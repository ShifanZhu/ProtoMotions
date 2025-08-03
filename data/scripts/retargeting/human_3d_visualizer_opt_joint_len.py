import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# ======================== #
# 1. SKELETON DEFINITIONS  #
# ======================== #

JOINT_NAMES = [
    "pelvis", "spine_top", "neck_top", "head_top",
    "right_shoulder", "right_elbow", "right_hand",
    "left_shoulder", "left_elbow", "left_hand",
    "right_hip", "right_knee", "right_foot",
    "left_hip", "left_knee", "left_foot"
]
JOINT_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}
(
    PELVIS, SPINE_TOP, NECK_TOP, HEAD_TOP,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HAND,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_HAND,
    RIGHT_HIP, RIGHT_KNEE, RIGHT_FOOT,
    LEFT_HIP, LEFT_KNEE, LEFT_FOOT
) = range(16)

def DOF(joint, axis): return 3 * joint + axis
SPINE_TOP_YAW = DOF(SPINE_TOP, 0)
RIGHT_SHOULDER_YAW, RIGHT_SHOULDER_PITCH, RIGHT_SHOULDER_ROLL = DOF(RIGHT_SHOULDER, 0), DOF(RIGHT_SHOULDER, 1), DOF(RIGHT_SHOULDER, 2)
RIGHT_ELBOW_YAW, RIGHT_ELBOW_PITCH, RIGHT_ELBOW_ROLL = DOF(RIGHT_ELBOW, 0), DOF(RIGHT_ELBOW, 1), DOF(RIGHT_ELBOW, 2)
LEFT_SHOULDER_YAW, LEFT_SHOULDER_PITCH, LEFT_SHOULDER_ROLL = DOF(LEFT_SHOULDER, 0), DOF(LEFT_SHOULDER, 1), DOF(LEFT_SHOULDER, 2)
LEFT_ELBOW_YAW, LEFT_ELBOW_PITCH, LEFT_ELBOW_ROLL = DOF(LEFT_ELBOW, 0), DOF(LEFT_ELBOW, 1), DOF(LEFT_ELBOW, 2)
RIGHT_HIP_YAW, RIGHT_HIP_PITCH, RIGHT_HIP_ROLL = DOF(RIGHT_HIP, 0), DOF(RIGHT_HIP, 1), DOF(RIGHT_HIP, 2)
RIGHT_KNEE_YAW, RIGHT_KNEE_PITCH, RIGHT_KNEE_ROLL = DOF(RIGHT_KNEE, 0), DOF(RIGHT_KNEE, 1), DOF(RIGHT_KNEE, 2)
LEFT_HIP_YAW, LEFT_HIP_PITCH, LEFT_HIP_ROLL = DOF(LEFT_HIP, 0), DOF(LEFT_HIP, 1), DOF(LEFT_HIP, 2)
LEFT_KNEE_YAW, LEFT_KNEE_PITCH, LEFT_KNEE_ROLL = DOF(LEFT_KNEE, 0), DOF(LEFT_KNEE, 1), DOF(LEFT_KNEE, 2)

BONES_IDX = [
    (PELVIS, SPINE_TOP), (SPINE_TOP, NECK_TOP), (NECK_TOP, HEAD_TOP),
    (SPINE_TOP, RIGHT_SHOULDER), (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_HAND),
    (SPINE_TOP, LEFT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_HAND),
    (PELVIS, RIGHT_HIP), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_FOOT),
    (PELVIS, LEFT_HIP), (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_FOOT)
]

BONE_LENGTHS = {
    'spine': 0.5, 'neck': 0.1, 'head': 0.1,
    'upper_arm': 0.3, 'lower_arm': 0.25,
    'upper_leg': 0.4, 'lower_leg': 0.4,
    'shoulder_offset': 0.2, 'hip_offset': 0.1
}

# Keys of bone lengths to optimize
BONE_LENGTH_KEYS_TO_OPTIMIZE = ['upper_arm', 'lower_arm', 'upper_leg', 'lower_leg']

def get_default_bone_to_optimize_lengths_vec():
    return np.array([BONE_LENGTHS[key] for key in BONE_LENGTH_KEYS_TO_OPTIMIZE])

def update_bone_lengths_from_vec(bone_lengths, vec):
    for k, v in zip(BONE_LENGTH_KEYS_TO_OPTIMIZE, vec):
        bone_lengths[k] = v
    return bone_lengths

DEFAULT_OPT_JOINTS = {
    RIGHT_HAND:  [SPINE_TOP_YAW, RIGHT_SHOULDER_YAW, RIGHT_SHOULDER_PITCH, RIGHT_SHOULDER_ROLL,
                  RIGHT_ELBOW_YAW, RIGHT_ELBOW_PITCH, RIGHT_ELBOW_ROLL],
    LEFT_HAND:   [SPINE_TOP_YAW, LEFT_SHOULDER_YAW, LEFT_SHOULDER_PITCH, LEFT_SHOULDER_ROLL,
                  LEFT_ELBOW_YAW, LEFT_ELBOW_PITCH, LEFT_ELBOW_ROLL],
    RIGHT_ELBOW: [SPINE_TOP_YAW, RIGHT_SHOULDER_YAW, RIGHT_SHOULDER_PITCH, RIGHT_SHOULDER_ROLL,
                  RIGHT_ELBOW_YAW, RIGHT_ELBOW_PITCH, RIGHT_ELBOW_ROLL],
    LEFT_ELBOW:  [SPINE_TOP_YAW, LEFT_SHOULDER_YAW, LEFT_SHOULDER_PITCH, LEFT_SHOULDER_ROLL,
                  LEFT_ELBOW_YAW, LEFT_ELBOW_PITCH, LEFT_ELBOW_ROLL],
    RIGHT_FOOT:  [RIGHT_HIP_YAW, RIGHT_HIP_PITCH, RIGHT_HIP_ROLL,
                  RIGHT_KNEE_YAW, RIGHT_KNEE_PITCH, RIGHT_KNEE_ROLL],
    LEFT_FOOT:   [LEFT_HIP_YAW, LEFT_HIP_PITCH, LEFT_HIP_ROLL,
                  LEFT_KNEE_YAW, LEFT_KNEE_PITCH, LEFT_KNEE_ROLL],
    RIGHT_KNEE:  [RIGHT_HIP_YAW, RIGHT_HIP_PITCH, RIGHT_HIP_ROLL,
                  RIGHT_KNEE_YAW, RIGHT_KNEE_PITCH, RIGHT_KNEE_ROLL],
    LEFT_KNEE:   [LEFT_HIP_YAW, LEFT_HIP_PITCH, LEFT_HIP_ROLL,
                  LEFT_KNEE_YAW, LEFT_KNEE_PITCH, LEFT_KNEE_ROLL],
}

def get_default_joint_angles():
    return np.zeros(48)

def rot_x(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def get_joint_positions_and_orientations(bone_lengths, joint_angles):
    # Unpack bone lengths
    spine_len, neck_len, head_len = bone_lengths['spine'], bone_lengths['neck'], bone_lengths['head']
    upper_arm_len, lower_arm_len = bone_lengths['upper_arm'], bone_lengths['lower_arm']
    upper_leg_len, lower_leg_len = bone_lengths['upper_leg'], bone_lengths['lower_leg']
    shoulder_offset, hip_offset = bone_lengths['shoulder_offset'], bone_lengths['hip_offset']

    def ang(idx): return joint_angles[3*idx:3*idx+3]
    joint_positions, joint_orientations = [], []

    p = np.array([0., 0., 0.])
    R = rot_z(ang(0)[0]) @ rot_y(ang(0)[1]) @ rot_x(ang(0)[2])
    joint_positions.append(p); joint_orientations.append(R)

    R_spine = R @ rot_z(ang(1)[0]) @ rot_y(ang(1)[1]) @ rot_x(ang(1)[2])
    p_spine = p + R @ np.array([0, 0, spine_len])
    joint_positions.append(p_spine); joint_orientations.append(R_spine)

    R_neck = R_spine @ rot_z(ang(2)[0]) @ rot_y(ang(2)[1]) @ rot_x(ang(2)[2])
    p_neck = p_spine + R_spine @ np.array([0, 0, neck_len])
    joint_positions.append(p_neck); joint_orientations.append(R_neck)

    R_head = R_neck @ rot_z(ang(3)[0]) @ rot_y(ang(3)[1]) @ rot_x(ang(3)[2])
    p_head = p_neck + R_neck @ np.array([0, 0, head_len])
    joint_positions.append(p_head); joint_orientations.append(R_head)

    R_sho_r = R_spine @ rot_z(ang(4)[0]) @ rot_y(ang(4)[1]) @ rot_x(ang(4)[2])
    p_sho_r = p_spine + R_spine @ np.array([0, -shoulder_offset, 0])
    joint_positions.append(p_sho_r); joint_orientations.append(R_sho_r)
    R_elb_r = R_sho_r @ rot_z(ang(5)[0]) @ rot_y(ang(5)[1]) @ rot_x(ang(5)[2])
    p_elb_r = p_sho_r + R_sho_r @ np.array([0, -upper_arm_len, 0])
    joint_positions.append(p_elb_r); joint_orientations.append(R_elb_r)
    R_hand_r = R_elb_r @ rot_z(ang(6)[0]) @ rot_y(ang(6)[1]) @ rot_x(ang(6)[2])
    p_hand_r = p_elb_r + R_elb_r @ np.array([0, -lower_arm_len, 0])
    joint_positions.append(p_hand_r); joint_orientations.append(R_hand_r)

    R_sho_l = R_spine @ rot_z(ang(7)[0]) @ rot_y(ang(7)[1]) @ rot_x(ang(7)[2])
    p_sho_l = p_spine + R_spine @ np.array([0, shoulder_offset, 0])
    joint_positions.append(p_sho_l); joint_orientations.append(R_sho_l)
    R_elb_l = R_sho_l @ rot_z(ang(8)[0]) @ rot_y(ang(8)[1]) @ rot_x(ang(8)[2])
    p_elb_l = p_sho_l + R_sho_l @ np.array([0, upper_arm_len, 0])
    joint_positions.append(p_elb_l); joint_orientations.append(R_elb_l)
    R_hand_l = R_elb_l @ rot_z(ang(9)[0]) @ rot_y(ang(9)[1]) @ rot_x(ang(9)[2])
    p_hand_l = p_elb_l + R_elb_l @ np.array([0, upper_arm_len, 0])
    joint_positions.append(p_hand_l); joint_orientations.append(R_hand_l)

    R_hip_r = R @ rot_z(ang(10)[0]) @ rot_y(ang(10)[1]) @ rot_x(ang(10)[2])
    p_hip_r = p + R @ np.array([0, -hip_offset, 0])
    joint_positions.append(p_hip_r); joint_orientations.append(R_hip_r)
    R_knee_r = R_hip_r @ rot_z(ang(11)[0]) @ rot_y(ang(11)[1]) @ rot_x(ang(11)[2])
    p_knee_r = p_hip_r + R_hip_r @ np.array([0, 0, -upper_leg_len])
    joint_positions.append(p_knee_r); joint_orientations.append(R_knee_r)
    R_foot_r = R_knee_r @ rot_z(ang(12)[0]) @ rot_y(ang(12)[1]) @ rot_x(ang(12)[2])
    p_foot_r = p_knee_r + R_knee_r @ np.array([0, 0, -lower_leg_len])
    joint_positions.append(p_foot_r); joint_orientations.append(R_foot_r)

    R_hip_l = R @ rot_z(ang(13)[0]) @ rot_y(ang(13)[1]) @ rot_x(ang(13)[2])
    p_hip_l = p + R @ np.array([0, hip_offset, 0])
    joint_positions.append(p_hip_l); joint_orientations.append(R_hip_l)
    R_knee_l = R_hip_l @ rot_z(ang(14)[0]) @ rot_y(ang(14)[1]) @ rot_x(ang(14)[2])
    p_knee_l = p_hip_l + R_hip_l @ np.array([0, 0, -upper_leg_len])
    joint_positions.append(p_knee_l); joint_orientations.append(R_knee_l)
    R_foot_l = R_knee_l @ rot_z(ang(15)[0]) @ rot_y(ang(15)[1]) @ rot_x(ang(15)[2])
    p_foot_l = p_knee_l + R_knee_l @ np.array([0, 0, -lower_leg_len])
    joint_positions.append(p_foot_l); joint_orientations.append(R_foot_l)

    return np.vstack(joint_positions), joint_orientations

# ==========================================
# Multi-Target Inverse Kinematics with Bone Lengths
# ==========================================
def multi_target_ik_opt_bones(
    bone_lengths,
    joint_angles,
    targets,
    end_effector_idxs,
    opt_joint_indices_list=None,
    max_iters=100,
    tolerance=1e-2,
    step_size=1.0,
    step_size_bl=0.1,      # step for bone length update
    verbose=False
):
    joint_angles = np.array(joint_angles).copy()
    bone_to_optimize_lengths_vec = get_default_bone_to_optimize_lengths_vec() # bones to optimize lengths
    all_bone_lengths_cur = BONE_LENGTHS.copy() # all bone lengths
    n_joints = len(joint_angles)
    n_bones = len(bone_to_optimize_lengths_vec)

    if opt_joint_indices_list is None:
        opt_joint_indices_list = [
            DEFAULT_OPT_JOINTS.get(eff_idx, list(range(n_joints)))
            for eff_idx in end_effector_idxs
        ]
    if verbose:
        print("Optimizing joint indices for each target:")
        for i, (eff_idx, opt_idx) in enumerate(zip(end_effector_idxs, opt_joint_indices_list)):
            eff_name = JOINT_NAMES[eff_idx] if eff_idx < len(JOINT_NAMES) else f"Index {eff_idx}"
            joint_names = [JOINT_NAMES[j//3] + f"_{['yaw','pitch','roll'][j%3]}" if (j//3) < len(JOINT_NAMES) else str(j)
                          for j in opt_idx]
            print(f"  Target {i}: {eff_name} (index {eff_idx})")
            print(f"    Optimizing DOFs: {joint_names}")

    angles_history = [joint_angles.copy()]
    bone_length_history = [bone_to_optimize_lengths_vec.copy()]

    for it in range(max_iters):
        all_bone_lengths_cur = update_bone_lengths_from_vec(BONE_LENGTHS.copy(), bone_to_optimize_lengths_vec)
        joint_positions, _ = get_joint_positions_and_orientations(all_bone_lengths_cur, joint_angles)
        error_vecs = []
        d_theta_accum = np.zeros(n_joints)
        d_theta_count = np.zeros(n_joints)
        d_bl_accum = np.zeros(n_bones)
        d_bl_count = np.zeros(n_bones)

        for t, (target, eff_idx, opt_joint_indices) in enumerate(zip(targets, end_effector_idxs, opt_joint_indices_list)):
            # print(f"Target {t+1}/{len(targets)}: {JOINT_NAMES[eff_idx]} at {target}")
            eff_pos = joint_positions[eff_idx]
            error = target - eff_pos
            error_vecs.append(error)
            # Joint angle Jacobian
            J_theta = np.zeros((3, len(opt_joint_indices)))
            delta_theta = 1e-2
            for j, idx in enumerate(opt_joint_indices):
                orig = joint_angles[idx]
                joint_angles[idx] += delta_theta
                joint_positions_pert, _ = get_joint_positions_and_orientations(all_bone_lengths_cur, joint_angles)
                eff_pos_pert = joint_positions_pert[eff_idx]
                J_theta[:, j] = (eff_pos_pert - eff_pos) / delta_theta
                joint_angles[idx] = orig
            d_theta = step_size * J_theta.T @ error
            for j, idx in enumerate(opt_joint_indices):
                d_theta_accum[idx] += d_theta[j]
                d_theta_count[idx] += 1

            # Bone length Jacobian
            J_bl = np.zeros((3, n_bones))
            delta_bl = 1e-2
            for b, key in enumerate(BONE_LENGTH_KEYS_TO_OPTIMIZE):
                orig = bone_to_optimize_lengths_vec[b]
                bone_to_optimize_lengths_vec[b] += delta_bl
                bone_lengths_tmp = update_bone_lengths_from_vec(BONE_LENGTHS.copy(), bone_to_optimize_lengths_vec)
                joint_positions_pert, _ = get_joint_positions_and_orientations(bone_lengths_tmp, joint_angles)
                eff_pos_pert = joint_positions_pert[eff_idx]
                J_bl[:, b] = (eff_pos_pert - eff_pos) / delta_bl
                bone_to_optimize_lengths_vec[b] = orig
            d_bl = step_size_bl * J_bl.T @ error
            d_bl_accum += d_bl
            d_bl_count += 1

        update_indices = np.where(d_theta_count > 0)[0] # update specific joints
        joint_angles[update_indices] += d_theta_accum[update_indices] / d_theta_count[update_indices]

        update_bl_indices = np.where(d_bl_count > 0)[0]
        bone_to_optimize_lengths_vec[update_bl_indices] += d_bl_accum[update_bl_indices] / d_bl_count[update_bl_indices]
        bone_to_optimize_lengths_vec = np.clip(bone_to_optimize_lengths_vec, 0.05, 2.0)

        angles_history.append(joint_angles.copy())
        bone_length_history.append(bone_to_optimize_lengths_vec.copy())

        err_norm = np.linalg.norm(np.concatenate(error_vecs))
        if verbose:
            print(f"Iteration {it+1}: Total error norm = {err_norm:.5f}, bone_lengths = {bone_to_optimize_lengths_vec}")
        if err_norm < tolerance:
            if verbose:
                print(f"IK converged in {it} iterations. Final error: {err_norm:.5f}")
            break

    bone_lengths_final = update_bone_lengths_from_vec(BONE_LENGTHS.copy(), bone_to_optimize_lengths_vec)
    return joint_angles, bone_lengths_final, angles_history, bone_length_history

# ================================== #
# 4. VISUALIZATION UTILS & EXECUTION #
# ================================== #

def draw_frame(ax, origin, R, length=0.05):
    x_axis = origin + R @ np.array([length, 0, 0])
    y_axis = origin + R @ np.array([0, length, 0])
    z_axis = origin + R @ np.array([0, 0, length])
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r', linewidth=3)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g', linewidth=3)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b', linewidth=3)

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_skeleton(
    ax, joint_positions, joint_orientations,
    targets=None, target_names=None, show_axes=True, title=''
):
    ax.clear()
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], color='red', s=50)
    if targets is not None:
        n_targets = len(targets)
        colors = plt.cm.get_cmap('tab10', n_targets)
        for idx, target in enumerate(targets):
            name = target_names[idx] if (target_names is not None and idx < len(target_names)) else f'Target {idx+1}'
            ax.scatter(*target, color=colors(idx), s=150, label=name)
    for b in BONES_IDX:
        xs, ys, zs = zip(*joint_positions[list(b)])
        ax.plot(xs, ys, zs, color='black', linewidth=2)
    for i, (pos, R) in enumerate(zip(joint_positions, joint_orientations)):
        if show_axes:
            draw_frame(ax, pos, R, length=0.05)
        ax.text(pos[0], pos[1], pos[2], f'{i}: {JOINT_NAMES[i]}', color='darkblue', fontsize=9)
    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left)')
    ax.set_zlabel('Z (up)')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    ax.legend(loc='upper right')

# ================ #
# 5. MAIN EXECUTE  #
# ================ #

if __name__ == "__main__":
    joint_angles = get_default_joint_angles()
    targets = [
        np.array([0.5, -0.3, 0.8]),  # right hand
        np.array([0.5, 0.3, 0.8]),   # left hand
        np.array([0.2, 0.3, 0.0]),   # left foot
        np.array([-0.2, -0.2, 0.0]), # right foot
    ]
    end_effector_idxs = [RIGHT_HAND, LEFT_HAND, LEFT_FOOT, RIGHT_FOOT]
    target_names = [JOINT_NAMES[idx].replace("_", " ").title() for idx in end_effector_idxs]

    start_time = time.time()
    joint_angles_ik, bone_lengths_ik, angles_history, bone_length_history = multi_target_ik_opt_bones(
        BONE_LENGTHS, joint_angles, targets, end_effector_idxs, max_iters=150, step_size=0.5, step_size_bl=0.05, verbose=False
    )
    elapsed = time.time() - start_time
    print(f"IK optimization took {elapsed:.3f} seconds for {len(angles_history)} iterations.")
    print("Optimized bone lengths:", bone_lengths_ik)

    positions_history = []
    orientations_history = []
    for angles, bl_vec in zip(angles_history, bone_length_history):
        bone_lengths_this = update_bone_lengths_from_vec(BONE_LENGTHS.copy(), bl_vec)
        joint_positions, joint_orientations = get_joint_positions_and_orientations(bone_lengths_this, angles)
        positions_history.append(joint_positions)
        orientations_history.append(joint_orientations)
    
    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111, projection='3d')
    visualize_ik_iterations = False # Set True to see animation

    def animate(i):
        plot_skeleton(
            ax,
            positions_history[i],
            orientations_history[i],
            targets=targets,
            target_names=target_names,
            show_axes=True,
            title=f'IK Iteration {i+1}/{len(positions_history)}'
        )
    if visualize_ik_iterations:
        ani = FuncAnimation(fig, animate, frames=len(positions_history), interval=300, repeat=False)
    else:
        plot_skeleton(
            ax,
            positions_history[-1],
            orientations_history[-1],
            targets=targets,
            target_names=target_names,
            show_axes=True,
            title='3D Human Skeleton Visualization'
        )
        plt.tight_layout()
    plt.show()
