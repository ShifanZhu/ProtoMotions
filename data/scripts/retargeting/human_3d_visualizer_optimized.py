import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# ======================== #
# 1. SKELETON DEFINITIONS  #
# ======================== #

# JOINT INDEX CONSTANTS (auto-generated from list)
JOINT_NAMES = [
    "pelvis",          # 0
    "spine_top",       # 1
    "neck_top",        # 2
    "head_top",        # 3
    "right_shoulder",  # 4
    "right_elbow",     # 5
    "right_hand",      # 6
    "left_shoulder",   # 7
    "left_elbow",      # 8
    "left_hand",       # 9
    "right_hip",       # 10
    "right_knee",      # 11
    "right_foot",      # 12
    "left_hip",        # 13
    "left_knee",       # 14
    "left_foot",       # 15
]
# Build mapping from name to index
JOINT_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}
# For easier reading in code
(
    PELVIS, SPINE_TOP, NECK_TOP, HEAD_TOP,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HAND,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_HAND,
    RIGHT_HIP, RIGHT_KNEE, RIGHT_FOOT,
    LEFT_HIP, LEFT_KNEE, LEFT_FOOT
) = range(16)


# Map (joint, axis) to index in the flat joint_angles vector
# Axis: 0 = yaw, 1 = pitch, 2 = roll
def DOF(joint, axis): return 3 * joint + axis

# Named DOFs (for common limbs)
SPINE_TOP_YAW = DOF(SPINE_TOP, 0)
RIGHT_SHOULDER_YAW, RIGHT_SHOULDER_PITCH, RIGHT_SHOULDER_ROLL = DOF(RIGHT_SHOULDER, 0), DOF(RIGHT_SHOULDER, 1), DOF(RIGHT_SHOULDER, 2)
RIGHT_ELBOW_YAW, RIGHT_ELBOW_PITCH, RIGHT_ELBOW_ROLL = DOF(RIGHT_ELBOW, 0), DOF(RIGHT_ELBOW, 1), DOF(RIGHT_ELBOW, 2)
LEFT_SHOULDER_YAW, LEFT_SHOULDER_PITCH, LEFT_SHOULDER_ROLL = DOF(LEFT_SHOULDER, 0), DOF(LEFT_SHOULDER, 1), DOF(LEFT_SHOULDER, 2)
LEFT_ELBOW_YAW, LEFT_ELBOW_PITCH, LEFT_ELBOW_ROLL = DOF(LEFT_ELBOW, 0), DOF(LEFT_ELBOW, 1), DOF(LEFT_ELBOW, 2)
RIGHT_HIP_YAW, RIGHT_HIP_PITCH, RIGHT_HIP_ROLL = DOF(RIGHT_HIP, 0), DOF(RIGHT_HIP, 1), DOF(RIGHT_HIP, 2)
RIGHT_KNEE_YAW, RIGHT_KNEE_PITCH, RIGHT_KNEE_ROLL = DOF(RIGHT_KNEE, 0), DOF(RIGHT_KNEE, 1), DOF(RIGHT_KNEE, 2)
LEFT_HIP_YAW, LEFT_HIP_PITCH, LEFT_HIP_ROLL = DOF(LEFT_HIP, 0), DOF(LEFT_HIP, 1), DOF(LEFT_HIP, 2)
LEFT_KNEE_YAW, LEFT_KNEE_PITCH, LEFT_KNEE_ROLL = DOF(LEFT_KNEE, 0), DOF(LEFT_KNEE, 1), DOF(LEFT_KNEE, 2)

# =========
# Bones map
# =========
BONES_IDX = [
    # pelvis->spine->neck->head
    (PELVIS, SPINE_TOP), (SPINE_TOP, NECK_TOP), (NECK_TOP, HEAD_TOP),  # spine
    # right arm
    (SPINE_TOP, RIGHT_SHOULDER), (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_HAND),
    # left arm
    (SPINE_TOP, LEFT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_HAND),
    # right leg
    (PELVIS, RIGHT_HIP), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_FOOT),
    # left leg
    (PELVIS, LEFT_HIP), (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_FOOT)
]

# ---- Bone lengths (meters) ----
BONE_LENGTHS = {
    'spine': 0.5, 'neck': 0.1, 'head': 0.1,
    'upper_arm': 0.3, 'lower_arm': 0.25,
    'upper_leg': 0.4, 'lower_leg': 0.4,
    'shoulder_offset': 0.2, 'hip_offset': 0.1
}


# -------------
# Optimization map (angle indices, using readable constants)
# -------------
DEFAULT_OPT_JOINTS = {
    RIGHT_HAND:  [SPINE_TOP_YAW,
                  RIGHT_SHOULDER_YAW, RIGHT_SHOULDER_PITCH, RIGHT_SHOULDER_ROLL,
                  RIGHT_ELBOW_YAW, RIGHT_ELBOW_PITCH, RIGHT_ELBOW_ROLL],
    LEFT_HAND:   [DOF(SPINE_TOP, 0),
                  LEFT_SHOULDER_YAW, LEFT_SHOULDER_PITCH, LEFT_SHOULDER_ROLL,
                  LEFT_ELBOW_YAW, LEFT_ELBOW_PITCH, LEFT_ELBOW_ROLL],
    RIGHT_ELBOW: [SPINE_TOP_YAW,
                  RIGHT_SHOULDER_YAW, RIGHT_SHOULDER_PITCH, RIGHT_SHOULDER_ROLL,
                  RIGHT_ELBOW_YAW, RIGHT_ELBOW_PITCH, RIGHT_ELBOW_ROLL],
    LEFT_ELBOW:  [DOF(SPINE_TOP, 0),
                  LEFT_SHOULDER_YAW, LEFT_SHOULDER_PITCH, LEFT_SHOULDER_ROLL,
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

# DEFAULT_OPT_JOINTS = {
#     6:  [3, 12, 13, 14, 15, 16, 17],      # right hand
#     9:  [3, 21, 22, 23, 24, 25, 26],      # left hand
#     5:  [3, 12, 13, 14, 15, 16, 17],      # right elbow
#     8:  [3, 21, 22, 23, 24, 25, 26],      # left elbow
#     12: [30, 31, 32, 33, 34, 35],         # right foot
#     15: [39, 40, 41, 42, 43, 44, 45],     # left foot
#     11: [30, 31, 32, 33, 34, 35],         # right knee
#     14: [39, 40, 41, 42, 43, 44],         # left knee
# }
print("DEFAULT_OPT_JOINTS:", DEFAULT_OPT_JOINTS)

# ---- Initial joint angles (in radians) for 16-joint skeleton x 3 angles each (yaw, pitch, roll) ----
joint_angles = [0.0] * (len(JOINT_NAMES) * 3)
joint_angles = np.array(joint_angles)

# Initial joint angles (all zeros)
def get_default_joint_angles():
    return np.zeros(48)

# ====================== #
# 2. FK/ROTATION UTILS   #
# ====================== #

def rot_x(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def get_joint_positions_and_orientations(bone_lengths, joint_angles):
    # Unpack
    spine_len, neck_len, head_len = bone_lengths['spine'], bone_lengths['neck'], bone_lengths['head']
    upper_arm_len, lower_arm_len = bone_lengths['upper_arm'], bone_lengths['lower_arm']
    upper_leg_len, lower_leg_len = bone_lengths['upper_leg'], bone_lengths['lower_leg']
    shoulder_offset = bone_lengths['shoulder_offset']
    hip_offset = bone_lengths['hip_offset']

    # Angles: each joint has (yaw, pitch, roll), so use slices of 3
    def ang(idx): return joint_angles[3*idx:3*idx+3]
    joint_positions = []
    joint_orientations = []

    # Pelvis
    p = np.array([0., 0., 0.])
    R = rot_z(ang(0)[0]) @ rot_y(ang(0)[1]) @ rot_x(ang(0)[2])
    joint_positions.append(p)
    joint_orientations.append(R)

    # Spine top
    R_spine = R @ rot_z(ang(1)[0]) @ rot_y(ang(1)[1]) @ rot_x(ang(1)[2])
    p_spine = p + R @ np.array([0, 0, spine_len])
    joint_positions.append(p_spine)
    joint_orientations.append(R_spine)

    # Neck top
    R_neck = R_spine @ rot_z(ang(2)[0]) @ rot_y(ang(2)[1]) @ rot_x(ang(2)[2])
    p_neck = p_spine + R_spine @ np.array([0, 0, neck_len])
    joint_positions.append(p_neck)
    joint_orientations.append(R_neck)

    # Head top
    R_head = R_neck @ rot_z(ang(3)[0]) @ rot_y(ang(3)[1]) @ rot_x(ang(3)[2])
    p_head = p_neck + R_neck @ np.array([0, 0, head_len])
    joint_positions.append(p_head)
    joint_orientations.append(R_head)

    # Right arm
    R_sho_r = R_spine @ rot_z(ang(4)[0]) @ rot_y(ang(4)[1]) @ rot_x(ang(4)[2])
    p_sho_r = p_spine + R_spine @ np.array([0, -shoulder_offset, 0])
    joint_positions.append(p_sho_r)
    joint_orientations.append(R_sho_r)
    R_elb_r = R_sho_r @ rot_z(ang(5)[0]) @ rot_y(ang(5)[1]) @ rot_x(ang(5)[2])
    p_elb_r = p_sho_r + R_sho_r @ np.array([0, -upper_arm_len, 0])
    joint_positions.append(p_elb_r)
    joint_orientations.append(R_elb_r)
    R_hand_r = R_elb_r @ rot_z(ang(6)[0]) @ rot_y(ang(6)[1]) @ rot_x(ang(6)[2])
    p_hand_r = p_elb_r + R_elb_r @ np.array([0, -lower_arm_len, 0])
    joint_positions.append(p_hand_r)
    joint_orientations.append(R_hand_r)

    # Left arm
    R_sho_l = R_spine @ rot_z(ang(7)[0]) @ rot_y(ang(7)[1]) @ rot_x(ang(7)[2])
    p_sho_l = p_spine + R_spine @ np.array([0, shoulder_offset, 0])
    joint_positions.append(p_sho_l)
    joint_orientations.append(R_sho_l)
    R_elb_l = R_sho_l @ rot_z(ang(8)[0]) @ rot_y(ang(8)[1]) @ rot_x(ang(8)[2])
    p_elb_l = p_sho_l + R_sho_l @ np.array([0, upper_arm_len, 0])
    joint_positions.append(p_elb_l)
    joint_orientations.append(R_elb_l)
    R_hand_l = R_elb_l @ rot_z(ang(9)[0]) @ rot_y(ang(9)[1]) @ rot_x(ang(9)[2])
    p_hand_l = p_elb_l + R_elb_l @ np.array([0, upper_arm_len, 0])
    joint_positions.append(p_hand_l)
    joint_orientations.append(R_hand_l)

    # Right leg
    R_hip_r = R @ rot_z(ang(10)[0]) @ rot_y(ang(10)[1]) @ rot_x(ang(10)[2])
    p_hip_r = p + R @ np.array([0, -hip_offset, 0])
    joint_positions.append(p_hip_r)
    joint_orientations.append(R_hip_r)
    R_knee_r = R_hip_r @ rot_z(ang(11)[0]) @ rot_y(ang(11)[1]) @ rot_x(ang(11)[2])
    p_knee_r = p_hip_r + R_hip_r @ np.array([0, 0, -upper_leg_len])
    joint_positions.append(p_knee_r)
    joint_orientations.append(R_knee_r)
    R_foot_r = R_knee_r @ rot_z(ang(12)[0]) @ rot_y(ang(12)[1]) @ rot_x(ang(12)[2])
    p_foot_r = p_knee_r + R_knee_r @ np.array([0, 0, -lower_leg_len])
    joint_positions.append(p_foot_r)
    joint_orientations.append(R_foot_r)

    # Left leg
    R_hip_l = R @ rot_z(ang(13)[0]) @ rot_y(ang(13)[1]) @ rot_x(ang(13)[2])
    p_hip_l = p + R @ np.array([0, hip_offset, 0])
    joint_positions.append(p_hip_l)
    joint_orientations.append(R_hip_l)
    R_knee_l = R_hip_l @ rot_z(ang(14)[0]) @ rot_y(ang(14)[1]) @ rot_x(ang(14)[2])
    p_knee_l = p_hip_l + R_hip_l @ np.array([0, 0, -upper_leg_len])
    joint_positions.append(p_knee_l)
    joint_orientations.append(R_knee_l)
    R_foot_l = R_knee_l @ rot_z(ang(15)[0]) @ rot_y(ang(15)[1]) @ rot_x(ang(15)[2])
    p_foot_l = p_knee_l + R_knee_l @ np.array([0, 0, -lower_leg_len])
    joint_positions.append(p_foot_l)
    joint_orientations.append(R_foot_l)

    return np.vstack(joint_positions), joint_orientations

# ==========================================
# Multi-Target Inverse Kinematics (multi-IK)
# ==========================================
def multi_target_ik(
    bone_lengths,
    joint_angles,
    targets,
    end_effector_idxs,
    opt_joint_indices_list=None,
    max_iters=100,
    tolerance=1e-2,
    step_size=1.0,
    verbose=False
):
    """
    Multi-end-effector IK where each target has its own set of joints to optimize.
    """
    joint_angles = np.array(joint_angles).copy()
    n_joints = len(joint_angles)

    # Use default mapping if no custom opt_joint_indices_list given
    if opt_joint_indices_list is None:
        opt_joint_indices_list = [
            DEFAULT_OPT_JOINTS.get(eff_idx, list(range(n_joints)))
            for eff_idx in end_effector_idxs
        ]
    if verbose:
        print("Optimizing these joint indices for each target:")
        for i, (eff_idx, opt_idx) in enumerate(zip(end_effector_idxs, opt_joint_indices_list)):
            print(f"  Target {i} (joint {eff_idx}): {opt_idx}")

    angles_history = [joint_angles.copy()]
    for it in range(max_iters):
        joint_positions, _ = get_joint_positions_and_orientations(bone_lengths, joint_angles)
        error_vecs = []
        d_theta_accum = np.zeros(n_joints)
        d_theta_count = np.zeros(n_joints)

        for target, eff_idx, opt_joint_indices in zip(targets, end_effector_idxs, opt_joint_indices_list):
            eff_pos = joint_positions[eff_idx]
            error = target - eff_pos
            error_vecs.append(error)
            J = np.zeros((3, len(opt_joint_indices)))
            delta = 1e-2
            for j, idx in enumerate(opt_joint_indices):
                orig = joint_angles[idx]
                joint_angles[idx] += delta
                joint_positions_pert, _ = get_joint_positions_and_orientations(bone_lengths, joint_angles)
                eff_pos_pert = joint_positions_pert[eff_idx]
                J[:, j] = (eff_pos_pert - eff_pos) / delta
                joint_angles[idx] = orig
            d_theta = step_size * J.T @ error
            for j, idx in enumerate(opt_joint_indices):
                d_theta_accum[idx] += d_theta[j]
                d_theta_count[idx] += 1

        # Apply average update to each joint that was used by any target
        update_indices = np.where(d_theta_count > 0)[0]
        joint_angles[update_indices] += d_theta_accum[update_indices] / d_theta_count[update_indices]
        angles_history.append(joint_angles.copy())

        # Total error norm for all targets
        err_norm = np.linalg.norm(np.concatenate(error_vecs))
        if verbose:
            print(f"Iteration {it+1}: Total error norm = {err_norm:.5f}")
        if err_norm < tolerance:
            if verbose:
                print(f"IK converged in {it} iterations. Final error: {err_norm:.5f}")
            break
    return joint_angles, angles_history

# ================================== #
# 4. VISUALIZATION UTILS & EXECUTION #
# ================================== #

def draw_frame(ax, origin, R, length=0.05):
    """Draw a coordinate frame at the given origin with rotation R."""
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

def plot_skeleton(ax, joint_positions, joint_orientations, targets=None, show_axes=True, title=''):
    ax.clear()
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], color='red', s=50)
    # Targets
    if targets is not None:
        for target in targets:
            ax.scatter(target[0], target[1], target[2], color='green', s=150, label='Target')
    # Bones
    for b in BONES_IDX:
        xs, ys, zs = zip(*joint_positions[list(b)])
        ax.plot(xs, ys, zs, color='black', linewidth=2)
    # Joint names & axes
    for i, (pos, R) in enumerate(zip(joint_positions, joint_orientations)):
        if show_axes: draw_frame(ax, pos, R, length=0.05)
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
        np.array([-0.2, -0.2, 0.0])  # right foot
    ]
    end_effector_idxs = [6, 9, 15, 12]  # indices in JOINT_NAMES

    start_time = time.time()
    joint_angles_ik, angles_history = multi_target_ik(
        BONE_LENGTHS, joint_angles, targets, end_effector_idxs, max_iters=150, step_size=0.5, verbose=True
    )
    elapsed = time.time() - start_time
    print(f"IK optimization took {elapsed:.3f} seconds for {len(angles_history)} iterations.")

    # Precompute all joint positions for each iteration for animation
    positions_history = []
    orientations_history = []
    for angles in angles_history:
        joint_positions, joint_orientations = get_joint_positions_and_orientations(BONE_LENGTHS, angles)
        positions_history.append(joint_positions)
        orientations_history.append(joint_orientations)
    
    fig = plt.figure(figsize=(7,9))
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        plot_skeleton(
            ax,
            positions_history[i],
            orientations_history[i],
            targets=targets,
            show_axes=True,
            title=f'IK Iteration {i+1}/{len(positions_history)}'
        )
    
    visualize_ik_iterations = False  # Set True to show animation

    if visualize_ik_iterations:
        ani = FuncAnimation(fig, animate, frames=len(positions_history), interval=300, repeat=False)
    else:
        # Show final result
        plot_skeleton(
            ax,
            positions_history[-1],
            orientations_history[-1],
            targets=targets,
            show_axes=True,
            title='3D Human Skeleton Visualization'
        )
        plt.tight_layout()
    plt.show()
