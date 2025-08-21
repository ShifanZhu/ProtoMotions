import typer
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import torch
from scipy.spatial.transform import Rotation as sRot
import uuid

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_joint_names import (
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)

import mink
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree

from tqdm import tqdm
from pathlib import Path
import csv


def save_all_marker_positions_txt(
    out_path: str | Path,
    global_translations: np.ndarray,   # [T, J, 3]
    names: list[str],                  # marker names (order must match the 2nd dim)
    fps: float,
    *,
    robot_type: str = "h1",
    scaled_for_viewer: bool = True,    # apply _RESCALE_FACTOR/_OFFSET like the viewer does
    layout: str = "wide",              # "wide" or "long"
    float_fmt: str = "%.6f",
) -> None:
    """
    Save marker positions to CSV (wide/long), skipping Toe/Wrist/all hand digits/Thorax.
    """
    import csv  # local import to keep dependency self-contained

    # --- 1) copy & (optionally) scale like the viewer ---
    gt = np.asarray(global_translations).copy()  # [T, J, 3]
    if scaled_for_viewer:
        scale = _RESCALE_FACTOR.get(robot_type, np.ones(3))
        gt = gt * scale[None, None, :]
        z_off = _OFFSET.get(robot_type, 0.0)
        gt[..., 2] += z_off

    # --- 2) filter out unwanted markers by name (case-insensitive contains) ---
    exclude_tokens = ("Toe", "Index", "Middle", "Pinky", "Ring", "Thumb", "Thorax")
    keep_idx = [i for i, n in enumerate(names)
                if not any(tok.lower() in n.lower() for tok in exclude_tokens)]
    if len(keep_idx) == 0:
        raise ValueError("After filtering, no markers remain to save. "
                         "Check your names or the exclusion list.")

    names_keep = [names[i] for i in keep_idx]
    gt = gt[:, keep_idx, :]  # [T, J_keep, 3]

    # --- 3) prepare output ---
    T, J_keep, _ = gt.shape
    times = np.arange(T, dtype=float) / float(fps)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 4) write in desired layout ---
    if layout.lower() == "wide":
        # One row per frame, columns: frame,time, <name1>_x,<name1>_y,<name1>_z, ...
        header = ["frame", "time_s"]
        for n in names_keep:
            header += [f"{n}_x", f"{n}_y", f"{n}_z"]

        mat = np.zeros((T, 2 + 3 * J_keep), dtype=float)
        mat[:, 0] = np.arange(T)
        mat[:, 1] = times
        mat[:, 2:] = gt.reshape(T, 3 * J_keep)

        np.savetxt(
            out_path,
            mat,
            delimiter=",",
            fmt=float_fmt,
            header=",".join(header),
            comments="",
        )
    else:
        # "long" format: frame,time,name,x,y,z
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "name", "x", "y", "z"])
            for t in range(T):
                for j, name in enumerate(names_keep):
                    x, y, z = gt[t, j]
                    w.writerow([t, f"{times[t]:.6f}", name,
                                float_fmt % x, float_fmt % y, float_fmt % z])

@dataclass
class KeyCallback:
    pause: bool = False
    first_pose_only: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_SPACE:
            self.pause = not self.pause
        elif key == user_input.KEY_ENTER:
            self.first_pose_only = not self.first_pose_only
            print(f"First pose only: {self.first_pose_only}")


_HERE = Path(__file__).parent

_HAND_NAMES = ["Index", "Middle", "Pinky", "Ring", "Thumb", "Wrist"]
_IMPORTANT_NAMES = ["Shoulder", "Knee", "Toe", "Elbow", "Head"]

# Each entry, e.g., "L_Elbow": {"name": "left_elbow_link", ...} says:
# Use the mocap marker named "L_Elbow" as the target for the robot’s "left_elbow_link" frame.
# The mapping assumes that the marker's 3D position is very close to the actual elbow joint position
# on both the mocap subject and the robot.
_H1_KEYPOINT_TO_JOINT = {
    # We provide higher weight to the "end of graph nodes" as they are more important for recovering the overall motion
    "Head": {"name": "head", "weight": 3.0},
    "Pelvis": {"name": "pelvis", "weight": 1.0},
    "L_Hip": {"name": "left_hip_yaw_link", "weight": 1.0},
    "R_Hip": {"name": "right_hip_yaw_link", "weight": 1.0},
    "L_Knee": {"name": "left_knee_link", "weight": 1.0},
    "R_Knee": {"name": "right_knee_link", "weight": 1.0},
    "L_Ankle": {"name": "left_ankle_link", "weight": 3.0},
    "R_Ankle": {"name": "right_ankle_link", "weight": 3.0},
    "L_Toe": {"name": "left_foot_link", "weight": 3.0},
    "R_Toe": {"name": "right_foot_link", "weight": 3.0},
    "L_Elbow": {"name": "left_elbow_link", "weight": 1.0},
    "R_Elbow": {"name": "right_elbow_link", "weight": 1.0},
    "L_Wrist": {"name": "left_arm_end_effector", "weight": 3.0},
    "R_Wrist": {"name": "right_arm_end_effector", "weight": 3.0},
    "L_Shoulder": {"name": "left_shoulder_pitch_link", "weight": 1.0},
    "R_Shoulder": {"name": "right_shoulder_pitch_link", "weight": 1.0},
}

_G1_KEYPOINT_TO_JOINT = {
    "Pelvis": {"name": "pelvis", "weight": 1.0},
    "Head": {"name": "head", "weight": 1.0},
    # Legs.
    "L_Hip": {"name": "left_hip_yaw_link", "weight": 1.0},
    "R_Hip": {"name": "right_hip_yaw_link", "weight": 1.0},
    "L_Knee": {"name": "left_knee_link", "weight": 1.0},
    "R_Knee": {"name": "right_knee_link", "weight": 1.0},
    "L_Ankle": {"name": "left_ankle_roll_link", "weight": 1.0},
    "R_Ankle": {"name": "right_ankle_roll_link", "weight": 1.0},
    # Arms.
    "L_Elbow": {"name": "left_elbow_link", "weight": 1.0},
    "R_Elbow": {"name": "right_elbow_link", "weight": 1.0},
    "L_Wrist": {"name": "left_wrist_yaw_link", "weight": 1.0},
    "R_Wrist": {"name": "right_wrist_yaw_link", "weight": 1.0},
    "L_Shoulder": {"name": "left_shoulder_pitch_link", "weight": 1.0},
    "R_Shoulder": {"name": "right_shoulder_pitch_link", "weight": 1.0},
}

_HUMAN1_KEYPOINT_TO_JOINT = {
    "Pelvis": {"name": "base", "weight": 1.0},
    "Head": {"name": "Head", "weight": 1.0},
    # Legs.
    "L_Hip": {"name": "LeftUpperLeg", "weight": 1.0},
    "R_Hip": {"name": "RightUpperLeg", "weight": 1.0},
    "L_Knee": {"name": "LeftLowerLeg", "weight": 1.0},
    "R_Knee": {"name": "RightLowerLeg", "weight": 1.0},
    "L_Ankle": {"name": "LeftFoot", "weight": 1.0},
    "R_Ankle": {"name": "RightFoot", "weight": 1.0},
    # # Arms.
    "L_Elbow": {"name": "LeftForeArm", "weight": 1.0},
    "R_Elbow": {"name": "RightForeArm", "weight": 1.0},
    "L_Wrist": {"name": "LeftHand", "weight": 1.0},
    "R_Wrist": {"name": "RightHand", "weight": 1.0},
    "L_Shoulder": {"name": "LeftUpperArm", "weight": 1.0},
    "R_Shoulder": {"name": "RightUpperArm", "weight": 1.0},
}

_KEYPOINT_TO_JOINT_MAP = {
    "h1": _H1_KEYPOINT_TO_JOINT,
    "g1": _G1_KEYPOINT_TO_JOINT,
    "male_human_model": _HUMAN1_KEYPOINT_TO_JOINT
}

_RESCALE_FACTOR = {
    "h1": np.array([1.0, 1.0, 1.1]),
    "g1": np.array([0.75, 1.0, 0.8]),
    "male_human_model": np.array([1.0, 1.0, 1.0]),
}

_OFFSET = {
    "h1": 0.0,
}

_ROOT_LINK = {
    "h1": "pelvis",
    "g1": "pelvis",
    "male_human_model": "base",
}

_H1_VELOCITY_LIMITS = {
    "left_hip_yaw_joint": 23,
    "left_hip_roll_joint": 23,
    "left_hip_pitch_joint": 23,
    "left_knee_joint": 14,
    "left_ankle_joint": 9,
    "right_hip_yaw_joint": 23,
    "right_hip_roll_joint": 23,
    "right_hip_pitch_joint": 23,
    "right_knee_joint": 14,
    "right_ankle_joint": 9,
    "torso_joint": 23,
    "left_shoulder_pitch_joint": 9,
    "left_shoulder_roll_joint": 9,
    "left_shoulder_yaw_joint": 20,
    "left_elbow_joint": 20,
    "right_shoulder_pitch_joint": 9,
    "right_shoulder_roll_joint": 9,
    "right_shoulder_yaw_joint": 20,
    "right_elbow_joint": 20,
}

_VEL_LIMITS = {
    "h1": _H1_VELOCITY_LIMITS,
}


def construct_mj_model(robot_name: str, keypoint_names: Sequence[str]):
    root = mjcf.RootElement()

    root.visual.headlight.ambient = ".4 .4 .4"
    root.visual.headlight.diffuse = ".8 .8 .8"
    root.visual.headlight.specular = "0.1 0.1 0.1"
    root.visual.rgba.haze = "0 0 0 0"
    root.visual.quality.shadowsize = "8192"

    # 4k resolution.
    getattr(root.visual, "global").offheight = "2160"
    getattr(root.visual, "global").offwidth = "3840"

    root.asset.add(
        "texture",
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1="0 0 0",
        rgb2="0 0 0",
        width="800",
        height="800",
    )
    root.asset.add(
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0 0 0",
        rgb2="0 0 0",
        width="300",
        height="300",
        mark="edge",
        markrgb=".2 .3 .4",
    )
    root.asset.add(
        "material",
        name="grid",
        texture="grid",
        texrepeat="1 1",
        texuniform="true",
        reflectance=".2",
    )
    root.worldbody.add(
        "geom", name="ground", type="plane", size="0 0 .01", material="grid"
    )

    for keypoint_name in keypoint_names:
        if any(hand_name in keypoint_name for hand_name in _HAND_NAMES):
            size = 0.008
        else:
            size = 0.02
        body = root.worldbody.add(
            "body", name=f"keypoint_{keypoint_name}", mocap="true"
        )
        rgb = np.random.rand(3)
        body.add(
            "site",
            name=f"site_{keypoint_name}",
            type="sphere",
            size=f"{size}",
            rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1",
        )
        if keypoint_name == "Pelvis":
            body.add("light", pos="0 0 2", directional="false")
            root.worldbody.add(
                "camera",
                name="tracking01",
                pos=[2.972, -0.134, 1.303],
                xyaxes="0.294 0.956 0.000 -0.201 0.062 0.978",
                mode="trackcom",
            )
            root.worldbody.add(
                "camera",
                name="tracking02",
                pos="4.137 2.642 1.553",
                xyaxes="-0.511 0.859 0.000 -0.123 -0.073 0.990",
                mode="trackcom",
            )

    if robot_name == "h1":
        humanoid_mjcf = mjcf.from_path("protomotions/data/assets/mjcf/h1.xml")
    elif robot_name == "g1":
        humanoid_mjcf = mjcf.from_path("protomotions/data/assets/mjcf/g1.xml")
    elif robot_name == "male_human_model":
        humanoid_mjcf = mjcf.from_path("protomotions/data/assets/mjcf/male_human_model.xml")
    else:
        raise ValueError(f"Unknown robot name: {robot_name}")
    humanoid_mjcf.worldbody.add(
        "camera",
        name="front_track",
        pos="-0.120 3.232 1.064",
        xyaxes="-1.000 -0.002 -0.000 0.000 -0.103 0.995",
        mode="trackcom",
    )
    root.include_copy(humanoid_mjcf)

    root_str = to_string(root, pretty=True)
    assets = get_assets(root)
    return mujoco.MjModel.from_xml_string(root_str, assets)


def to_string(
    root: mjcf.RootElement,
    precision: float = 17,
    zero_threshold: float = 0.0,
    pretty: bool = False,
) -> str:
    from lxml import etree

    xml_string = root.to_xml_string(precision=precision, zero_threshold=zero_threshold)
    root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))

    # Remove hashes from asset filenames.
    tags = ["mesh", "texture"]
    for tag in tags:
        assets = [
            asset
            for asset in root.find("asset").iter()
            if asset.tag == tag and "file" in asset.attrib
        ]
        for asset in assets:
            name, extension = asset.get("file").split(".")
            asset.set("file", ".".join((name[:-41], extension)))

    if not pretty:
        return etree.tostring(root, pretty_print=True).decode()

    # Remove auto-generated names.
    for elem in root.iter():
        for key in elem.keys():
            if key == "name" and "unnamed" in elem.get(key):
                elem.attrib.pop(key)

    # Get string from lxml.
    xml_string = etree.tostring(root, pretty_print=True)

    # Remove redundant attributes.
    xml_string = xml_string.replace(b' gravcomp="0"', b"")

    # Insert spaces between top-level elements.
    lines = xml_string.splitlines()
    newlines = []
    for line in lines:
        newlines.append(line)
        if line.startswith(b"  <"):
            if line.startswith(b"  </") or line.endswith(b"/>"):
                newlines.append(b"")
    newlines.append(b"")
    xml_string = b"\n".join(newlines)

    return xml_string.decode()


def get_assets(root: mjcf.RootElement) -> dict[str, bytes]:
    assets = {}
    for file, payload in root.get_assets().items():
        name, extension = file.split(".")
        assets[".".join((name[:-41], extension))] = payload
    return assets


def create_robot_motion(
    poses: np.ndarray, trans: np.ndarray, orig_global_trans: np.ndarray, mocap_fr: float, robot_type: str
) -> SkeletonMotion:
    """Create a SkeletonMotion for H1 robot from poses and translations.
    Args:
        poses: Joint angles from mujoco [N, num_dof] in proper ordering - groups of 3 hinge joints per joint
        trans: Root transform [N, 7] (pos + quat)
        orig_global_trans: Original global translations [N, num_joints, 3]
        mocap_fr: Motion capture framerate
    Returns:
        SkeletonMotion: Motion data in proper format for H1
    """
    from data.scripts.retargeting.torch_humanoid_batch import Humanoid_Batch
    from data.scripts.retargeting.config import get_config

    # Initialize H1 humanoid batch with config
    cfg = get_config(robot_type)
    humanoid_batch = Humanoid_Batch(cfg)

    # Convert poses to proper format
    B, seq_len = 1, poses.shape[0]

    # Convert to tensor format
    poses_tensor = torch.from_numpy(poses).float().reshape(B, seq_len, -1, 1)

    # Add root rotation from trans quaternion
    root_rot = sRot.from_quat(np.roll(trans[:, 3:7], -1)).as_rotvec()
    root_rot_tensor = torch.from_numpy(root_rot).float().reshape(B, seq_len, 1, 3)

    # Combine root rotation with joint poses
    poses_tensor = torch.cat(
        [
            root_rot_tensor,
            humanoid_batch.dof_axis * poses_tensor,
            torch.zeros((1, seq_len, len(cfg.extend_config), 3)),
        ],
        axis=2,
    )

    # Prepare root translation
    trans_tensor = torch.from_numpy(trans[:, :3]).float().reshape(B, seq_len, 3)

    # Perform forward kinematics
    motion_data = humanoid_batch.fk_batch(
        poses_tensor, trans_tensor, return_full=True, dt=1.0 / mocap_fr
    )

    # Convert back to proper kinematic structure
    fk_return_proper = humanoid_batch.convert_to_proper_kinematic(motion_data)

    # Get lowest heights for both original and retargeted motions
    orig_lowest_heights = torch.from_numpy(orig_global_trans[..., 2].min(axis=1))
    retarget_lowest_heights = (
        fk_return_proper.global_translation[..., 2].min(dim=-1).values
    )

    # Calculate height adjustment to match original motion's lowest points
    height_offset = (retarget_lowest_heights - orig_lowest_heights).unsqueeze(-1)

    # Adjust global translations to match original heights
    fk_return_proper.global_translation[..., 2] -= height_offset

    curr_motion = {
        k: v.squeeze().detach().cpu() if torch.is_tensor(v) else v
        for k, v in fk_return_proper.items()
    }
    return curr_motion


def create_skeleton_motion(
    poses: np.ndarray, # [N, 54] all robot joint orientations in rpy
    trans: np.ndarray, # [N, 7] root pos + quat
    skeleton_tree: SkeletonTree,
    orig_global_trans: np.ndarray,
    mocap_fr: float,
) -> SkeletonMotion:
    """Create a SkeletonMotion from poses and translations.
    Args:
        poses: Joint angles from mujoco [N, 153] - groups of 3 hinge joints per joint
        trans: Root transform [N, 7] (pos + quat)
        skeleton_tree: Skeleton tree for the model
        orig_global_trans: Original global translations [N, num_joints, 3]
        mocap_fr: Motion capture framerate
    """
    n_frames = poses.shape[0]
    pose_quat = np.zeros((n_frames, 51, 4))  # 51 joints, each with quaternion

    # Convert each joint's 3 hinge rotations to a single quaternion
    for i in range(51):  # 51 joints
        angles = poses[
            :, i * 3 : (i + 1) * 3
        ]  # Get angles for this joint's x,y,z hinges
        pose_quat[:, i] = sRot.from_euler("XYZ", angles).as_quat()

    # Combine root orientation and all joint orientations
    full_pose = np.zeros((n_frames, 52, 4))  # 52 total joints (root + 51 joints)
    full_pose[:, 0] = np.roll(trans[:, 3:7], -1)  # Root quaternion
    full_pose[:, 1:] = pose_quat  # Other joint quaternions

    # Create skeleton state
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(full_pose), # all joints' orientation
        torch.from_numpy(trans[:, :3]), # root position
        is_local=True,
    )

    # Get global rotations and positions
    pose_quat_global = sk_state.global_rotation.numpy()
    global_pos = sk_state.global_translation.numpy()

    # Get lowest heights for both original and retargeted motions
    orig_lowest_heights = orig_global_trans[..., 2].min(axis=1, keepdims=True)
    retarget_lowest_heights = global_pos[..., 2].min(axis=1, keepdims=True)

    # Calculate height adjustment to match original motion's lowest points
    height_offset = retarget_lowest_heights - orig_lowest_heights

    # Adjust root translation to match original heights
    adjusted_trans = trans.copy()
    adjusted_trans[:, 2] -= height_offset.squeeze()

    # Create new skeleton state with adjusted heights
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        torch.from_numpy(adjusted_trans[:, :3]),
        is_local=False,
    )

    return SkeletonMotion.from_skeleton_state(new_sk_state, fps=mocap_fr)


# retarget_motion takes a SkeletonMotion (which encodes human-like joint positions/orientations for all frames)
# and a robot type, and computes a new sequence of robot joint angles such that the robot's key body parts best
# follow desired (possibly rescaled) 3D motion from mocap data—subject to the robot’s joint limits and constraints.
def retarget_motion(motion: SkeletonMotion, robot_type: str, render: bool = False):
    # Extract Motion Data
    # Loads 3D positions and orientations of all joints, for every frame, from the input motion.
    global_translations = motion.global_translation.numpy()
    pose_quat_global = motion.global_rotation.numpy()
    pose_quat = motion.local_rotation.numpy()
    timeseries_length = global_translations.shape[0]
    fps = motion.fps

    # Builds the MuJoCo robot model according to the specified type, including the world and robot skeleton.
    smplx_mujoco_joint_names = SMPLH_MUJOCO_NAMES
    mj_model = construct_mj_model(robot_type, smplx_mujoco_joint_names)
    # Sets up a mink.Configuration, an object to manage robot joint state for optimization.
    mink_configuration = mink.Configuration(mj_model)


    # Build Optimization Tasks
    # For each important body part (e.g., hand, ankle, head), create a task for the IK optimizer:
    # Try to make that robot body part match the corresponding mocap keypoint.
    # Each task has a cost (weight).
    # Add a posture task to regularize the robot’s pose.
    tasks = []

    frame_tasks = {}
    for joint_name, retarget_info in _KEYPOINT_TO_JOINT_MAP[robot_type].items():
        if robot_type == "h1":
            orientation_base_cost = 0
        else:
            orientation_base_cost = 0.0001
        task = mink.FrameTask(
            frame_name=retarget_info["name"],
            frame_type="body",
            position_cost=10.0 * retarget_info["weight"], # strongly encourage position matching
            orientation_cost=orientation_base_cost * retarget_info["weight"], # weakly encourage orientation matching
            lm_damping=1.0,
        )
        frame_tasks[retarget_info["name"]] = task # all tasks
    # "extend" adds each item from an iterable (like another list) to the end of the list, one by one.
    tasks.extend(frame_tasks.values())

    # The posture task typically penalizes deviation from a default or neutral pose: often the T-pose or the SMPL rest pose.
    posture_task = mink.PostureTask(mj_model, cost=1.0)
    tasks.append(posture_task)

    # Prepare MuJoCo model and data
    mj_model = mink_configuration.model
    data = mink_configuration.data

    key_callback = KeyCallback()

    # Modify the main processing loop to conditionally use the viewer
    # If visualization is enabled, launches a MuJoCo 3D viewer so you can watch the process.
    if render:
        viewer_context = mujoco.viewer.launch_passive(
            model=mj_model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=key_callback,
        )
    else:
        # Use contextlib.nullcontext as a no-op context manager
        from contextlib import nullcontext

        viewer_context = nullcontext()

    # store the optimized joint poses (qpos) and root translations for each frame,
    # so you can save/export the retargeted motion at the end.
    retargeted_poses = [] # all robot joint orientations in rpy
    retargeted_transformation = [] # root pos + quat

    # Main Frame-by-Frame IK Loop
    with viewer_context as viewer:
        if render:
            # Set up viewer camera only when rendering
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mj_model.cam("front_track").id

        # Directly set initial pose from first frame
        # Initialize qpos with zeros
        data.qpos[:] = 0

        # Set root position (first 3 values)
        data.qpos[0:3] = global_translations[0, 0]

        # Set root orientation (next 4 values)
        data.qpos[3:7] = pose_quat_global[0, 0]

        mink_configuration.update(data.qpos) # Updates mink configuration to match the new pose.
        mujoco.mj_forward(mj_model, data) # MuJoCo recomputes all positions, velocities, etc., given the new pose.
        posture_task.set_target_from_configuration(mink_configuration)
        mujoco.mj_step(mj_model, data) # advances the simulation by one step

        optimization_steps_per_frame = 2  # int(max(np.ceil(5.0 * 30 / fps), 1))
        rate = RateLimiter(frequency=fps * optimization_steps_per_frame)
        solver = "quadprog"

        t: int = int(np.ceil(-100.0 * fps / 30)) # t is from -100 to timeseries_length-1
        vel = None

        # Create progress bar
        pbar = tqdm(total=timeseries_length, desc="Retargeting frames")

        while (render and viewer.is_running() or not render) and t < timeseries_length:
            if not key_callback.pause:
                # === A. Set Target Positions and Orientations for This Frame ===
                for i, (joint_name, retarget_info) in enumerate(
                    _KEYPOINT_TO_JOINT_MAP[robot_type].items()
                ):
                    # When t < 0, we are in the initialization phase, so we use the first frame's data.
                    body_idx = smplx_mujoco_joint_names.index(joint_name)
                    target_pos = global_translations[max(0, t), body_idx, :].copy()

                    if robot_type in _RESCALE_FACTOR:
                        target_pos *= _RESCALE_FACTOR[robot_type]
                    if robot_type in _OFFSET:
                        target_pos[2] += _OFFSET[robot_type]

                    target_rot = pose_quat_global[max(0, t), body_idx].copy()
                    rot_matrix = sRot.from_quat(target_rot).as_matrix() # rotation in global frame
                    rot = mink.SO3.from_matrix(rot_matrix) # rotation in global frame
                    tasks[i].set_target(
                        mink.SE3.from_rotation_and_translation(rot, target_pos)
                    )

                # === B. Visualize MuJoCo Keypoint Markers (Ground truth)===
                keypoint_pos = {}
                for keypoint_name, keypoint in zip(
                    smplx_mujoco_joint_names, global_translations[max(0, t)]
                ):
                    mid = mj_model.body(f"keypoint_{keypoint_name}").mocapid[0]
                    data.mocap_pos[mid] = keypoint
                    keypoint_pos[keypoint_name] = keypoint

                # Perform multiple optimization steps
                # === C. Multiple IK Optimization Steps ===
                for _ in range(optimization_steps_per_frame):
                    limits = [
                        mink.ConfigurationLimit(mj_model), # joint limits
                    ]
                    if robot_type in _VEL_LIMITS and t >= 0:
                        limits.append(                     # velocity limits
                            mink.VelocityLimit(mj_model, _VEL_LIMITS[robot_type])
                        )
                    vel = mink.solve_ik(
                        mink_configuration, tasks, rate.dt, solver, 1e-1, limits=limits
                    )
                    mink_configuration.integrate_inplace(vel, rate.dt)
                    if render:
                        mujoco.mj_camlight(mj_model, data)

                # Store poses and translations if we're past initialization
                # === D. Save Results for This Frame ===
                if t >= 0:
                    retargeted_poses.append(data.qpos[7:].copy()) # [54] all robot joint orientations in rpy
                    retargeted_transformation.append(data.qpos[:7].copy()) # [7] root pos + quat
                    # print(f"Frame pose {t}:", data.qpos[7:].copy())
                    # print(f"Frame translation {t}:", data.qpos[:7].copy())

                if render and key_callback.first_pose_only and t == 0:
                    print(
                        "First pose set. Press Enter to continue animation, Space to pause/unpause"
                    )
                    key_callback.pause = True
                    key_callback.first_pose_only = False

                t += 1
                if t >= 0:  # Only update progress bar for actual frames
                    pbar.update(1)

            if render:
                viewer.sync()
                rate.sleep()

        pbar.close()

    # Convert stored motion to numpy arrays
    retargeted_poses = np.stack(retargeted_poses)
    retargeted_transformation = np.stack(retargeted_transformation)

    # Create skeleton motion
    if robot_type in ["h1", "g1"]:
        print("Creating robot motion for robot type:", robot_type)
        return create_robot_motion(
            retargeted_poses, retargeted_transformation, global_translations, fps, robot_type
        )
    else:
        print("Creating skeleton motion for robot type:", robot_type)
        skeleton_tree = SkeletonTree.from_mjcf(
            f"protomotions/data/assets/mjcf/{robot_type}.xml"
        )
        retargeted_motion = create_skeleton_motion(
            retargeted_poses, retargeted_transformation, skeleton_tree, global_translations, fps
        )

    return retargeted_motion


def manually_retarget_motion(
    amass_data: str, output_path: str, robot_type: str, render: bool = False
):
    # Store retargeted motion data into a dictionary
    motion_data = dict(np.load(open(amass_data, "rb"), allow_pickle=True))

    mujoco_joint_names = SMPLH_MUJOCO_NAMES
    joint_names = SMPLH_BONE_ORDER_NAMES

    # betas refers to the shape parameters of the body model,
    # which encodes the body shape variations across different individuals.
    betas = motion_data["betas"]
    gender = motion_data["gender"]
    amass_pose = motion_data["poses"] #! here pose is orientation
    amass_trans = motion_data["trans"] #! here trans is root joint (pelvis) positions
    if "mocap_framerate" in motion_data:
        mocap_fr = motion_data["mocap_framerate"]
    else:
        mocap_fr = motion_data["mocap_frame_rate"]

    skip = int(mocap_fr // 30)

    # Frames are downsampled to 30Hz for efficiency and consistency.
    pose_aa = torch.tensor(amass_pose[::skip]) # [N, 165] where N is number of frames
                                               # Refer to slides for SMPLH joint order
    amass_trans = torch.tensor(amass_trans[::skip]) # [N, 3]
    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

    # # # Add Gaussian noise to translation (trans)
    # # noise_std = 0.03  # You can adjust the noise level here
    # # amass_trans = amass_trans + noise_std * torch.randn_like(amass_trans)
    # # # Add Gaussian noise to pose_aa (joint angles)
    # # pose_noise_std = 0.03  # You can adjust the noise level here
    # # pose_aa = pose_aa + pose_noise_std * torch.randn_like(pose_aa)

    # # Extract sub-poses (each [N, 3])
    # global_orient = pose_aa[:, :3]
    # jaw_pose = pose_aa[:, 66:69]
    # leye_pose = pose_aa[:, 69:72]
    # reye_pose = pose_aa[:, 72:75]

    # body_pose = pose_aa[:, 3:66]
    # pelvis_pose = body_pose[:, :3]
    # L_hip_pose = body_pose[:, 3:6]
    # R_hip_pose = body_pose[:, 6:9]
    
    # # print("jaw_pose: ", jaw_pose)
    # # print("leye_pose: ", leye_pose)
    # # print("reye_pose: ", reye_pose)
    # # print("pelvis_pose: ", pelvis_pose)

    # N = pose_aa.shape[0]
    # t = np.arange(N)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(16, 8))

    # # Global orientation
    # plt.subplot(2, 2, 1)
    # plt.plot(t, global_orient.numpy())
    # plt.title('Global Orientation (axis-angle)')
    # plt.xlabel('Frame')
    # plt.ylabel('Value')
    # plt.legend(['x', 'y', 'z'])
    
    # # Pelvis pose
    # plt.subplot(2, 2, 2)
    # plt.plot(t, pelvis_pose.numpy())
    # plt.title('Pelvis Pose (axis-angle)')
    # plt.xlabel('Frame')
    # plt.ylabel('Value')
    # plt.legend(['x', 'y', 'z'])
    
    # # Left hip pose
    # plt.subplot(2, 2, 3)
    # plt.plot(t, L_hip_pose.numpy())
    # plt.title('Left Hip Pose (axis-angle)')
    # plt.xlabel('Frame')
    # plt.ylabel('Value')
    # plt.legend(['x', 'y', 'z'])
    
    # # # Right hip pose
    # # plt.subplot(2, 2, 4)
    # # plt.plot(t, R_hip_pose.numpy())
    # # plt.title('Right Hip Pose (axis-angle)')
    # # plt.xlabel('Frame')
    # # plt.ylabel('Value')
    # # plt.legend(['x', 'y', 'z'])
    
    # # amass_trans
    # plt.subplot(2, 2, 4)
    # plt.plot(t, amass_trans.numpy())
    # plt.title('AMASS Translation')
    # plt.xlabel('Frame')
    # plt.ylabel('Value')
    # plt.legend(['x', 'y', 'z'])

    # plt.tight_layout()
    # plt.show()


    betas = torch.from_numpy(betas)
    betas[:] = 0

    # Prepares the motion data dictionary.
    motion_data = {
        "pose_aa": pose_aa.numpy(), # remember this is axis-angle format
        "trans": amass_trans.numpy(), # root frame positions
        "beta": betas.numpy(),
    }

    # finds the index of q in joint_names and adds index to smpl_2_mujoco list
    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_joint_names if q in joint_names
    ]
    batch_size = motion_data["pose_aa"].shape[0] # whole number of frames

    pose_aa = np.concatenate( # [N, 156]
        [
            motion_data["pose_aa"][:, :66], # all body joints orientations
            motion_data["pose_aa"][:, 75:], # left/right hand orientations
                                            # skipping jaw and eyes, all zeros in amass data
        ],
        axis=-1,
    )
    # 52 = 22 body joints orientations + 15 left hand orientations + 15 right hand orientations
    # reshape from [N, 156] to [N, 52, 3] and reorder from smpl to mujoco joint order
    pose_aa_mj = pose_aa.reshape(batch_size, 52, 3)[:, smpl_2_mujoco]
    pose_quat = ( # change from axis-angle to quaternion [N, 52, 4]
        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 52, 4)
    )

    # A dictionary containing config parameters to build a humanoid model (based on SMPL or SMPL-X)
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": True,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smplx",
        "sim": "isaacgym",
    }
    # builds a robot model based on SMPL/SMPL-X
    smpl_local_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl", # SMPL model data
    )
    smpl_local_robot.load_from_skeleton(betas=betas[None,], gender=[0], objs_info=None)
    TMP_SMPL_DIR = "/tmp/smpl"
    uuid_str = uuid.uuid4()
    smpl_local_robot.write_xml(f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml")

    # Loads the skeleton tree from the generated MJCF XML file.
    # This skeleton tree represents the robot's joint hierarchy and structure.
    skeleton_tree = SkeletonTree.from_mjcf(
        f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
    )

    # Offsets mocap root position to match robot's reference frame. Purpose:
    # when transferring (retargeting) human motion capture (mocap) data onto a simulated robot 
    # (e.g., SMPL/SMPL-X in MuJoCo), the root position (the "pelvis" or base of the skeleton) 
    # in the mocap data may be in a different reference frame than the robot model in simulation. 
    # If not aligned, the robot could look "floating", "shifted", or misplaced.

    # skeleton_tree.local_translation[0] is the reference translation for the robot’s
    # root joint (e.g., "pelvis" or "base").

    root_trans_offset = ( # [N, 3] root positions for the robot's root joint after mocap data offset
        torch.from_numpy(motion_data["trans"]) # [N, 3] root positions from mocap data
        + skeleton_tree.local_translation[0]   # get [1, 3] from [52, 3]: reference translation for the robot's root joint
    )
    # print("Root translation shape:", skeleton_tree.local_translation.shape)
    # print("motion trans:", motion_data["trans"][0])
    # print("Root translation:", skeleton_tree.local_translation[0])
    # print("Root translation offset:", root_trans_offset[0])

    # Creates a temporary skeleton state from the pose and root translation.
    #* This is a temporary skeleton state just to compute global rotations.
    sk_state = SkeletonState.from_rotation_and_root_translation(
        # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
        skeleton_tree,               # The robot's skeleton structure (joint tree, hierarchy, etc.)
        torch.from_numpy(pose_quat), # The pose as a sequence of quaternions for all joints (local rotations)
        root_trans_offset,           # The root translation (with any necessary offset applied)
        is_local=True,               # Interpret `pose_quat` as **local joint rotations** (relative to parent joint)
    )
    # sk_state.local_rotation shape is [N, 52, 4], which is a joint relative to its parent joint
    # sk_state.global_rotation shape is [N, 52, 4], which is the absolute rotation in world space
    # sk_state.global_translation shape is [N, 52, 3], which is the absolute translation in world space


    # Transform All Rotations to a Global Reference
    # This step aligns the pose data’s global rotations with the robot’s coordinate conventions.
    timeseries_length = pose_aa.shape[0] # pose_aa is [N, 156], 156 = 52 joints * 3 axis-angle
    pose_quat_global = (
        (
            sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy())
            # This is a “conversion” quaternion, often used to switch between different quaternion conventions
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv() # without this, the robot will be rotated 90 degrees
        )
        .as_quat()
        .reshape(timeseries_length, -1, 4)
    )

    # Build a New SkeletonMotion Object with above calculated global rotations
    # Encodes the full trajectory (joint global orientations and global positions) for all N frames.
    # Builds a new skeleton state where the input rotations are interpreted as global (not local) rotations for each joint.
    # This lets you directly specify the pose in world coordinates, skipping the need for parent-to-child multiplication.
    trans = root_trans_offset.clone() # [N, 3] root positions for the robot's root joint after mocap data offset
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),  # now *global* rotations
        trans,
        is_local=False,                      # rotations are now *global*, not local
    )

    # Creates a SkeletonMotion object, which is a trajectory of poses, 
    # including joint orientations, root translations, etc., at a set framerate (here, 30 fps).
    new_sk_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30)
    
    # Retarget Motion (Key step)
    sk_motion = retarget_motion(new_sk_motion, robot_type, render=render)

    # Save the retargeted motion to the specified output path
    if robot_type in ["h1", "g1"]:
        torch.save(sk_motion, output_path)
    else:
        sk_motion.to_file(output_path)
    
    print(f"output_path: {output_path}")



    # Dump the mocap markers (the ones you visualize as keypoint_* bodies):
    save_all_marker_positions_txt(
        out_path=Path(output_path).with_suffix(".markers.txt"),
        global_translations=new_sk_motion.global_translation.numpy(),  # [T, J, 3]
        names=SMPLH_MUJOCO_NAMES,                                      # same order you used to build the world
        fps=new_sk_motion.fps,                                         # typically 30 after downsample
        robot_type=robot_type,
        scaled_for_viewer=True,    # False => raw AMASS/SMPL coords (no rescale/offset)
        layout="wide",             # or "long"
    )
    print("Saved marker positions to:", Path(output_path).with_suffix(".markers.txt"))



if __name__ == "__main__":
    typer.run(manually_retarget_motion)

# Example usage:
# python data/scripts/retargeting/mink_retarget.py data/amass/CNRS/283/01_L_1_stageii.npz data/output/test h1 --render
# python data/scripts/convert_amass_to_isaac.py data/amass/ --humanoid-type=smplx --robot-type=h1 --force-retarget