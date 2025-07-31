import numpy as np
import matplotlib.pyplot as plt

"""
This script visualizes a simplified 3D human skeleton using matplotlib, given bone lengths and joint angles.
It computes joint positions and local joint_orientations using forward kinematics, and plots the skeleton in 3D.

Joint Index Map:
----------------
  0: pelvis (root)
  1: spine top
  2: neck top
  3: head top
  4: right shoulder
  5: left shoulder
  6: right elbow
  7: right hand
  8: left elbow
  9: left hand
  10: right hip
  11: right knee
  12: right foot
  13: left hip
  14: left knee
  15: left foot

Link (Bone) Map:
----------------
  (0,1): pelvis -> spine top
  (1,2): spine top -> neck top
  (2,3): neck top -> head top
  (1,4): spine top -> right shoulder
  (4,6): right shoulder -> right elbow
  (6,7): right elbow -> right hand
  (1,5): spine top -> left shoulder
  (5,8): left shoulder -> left elbow
  (8,9): left elbow -> left hand
  (0,10): pelvis -> right hip
  (10,11): right hip -> right knee
  (11,12): right knee -> right foot
  (0,13): pelvis -> left hip
  (13,14): left hip -> left knee
  (14,15): left knee -> left foot
"""

bones = [
  (0,1), (1,2), (2,3),          # pelvis->neck->head
  (1,4), (4,6), (6,7),          # shoulderR->elbowR->handR
  (1,5), (5,8), (8,9),          # shoulderL->elbowL->handL
  (0,10), (10,11), (11,12),     # pelvis->hipR->kneeR->footR
  (0,13), (13,14), (14,15),     # pelvis->hipL->kneeL->footL
]

bone_lengths = {
  'spine': 0.5, 'neck': 0.1, 'head': 0.1,
  'upper_arm': 0.3, 'lower_arm': 0.25,
  'upper_leg': 0.4, 'lower_leg': 0.4,
}

axis_len = 0.1  # length of axis lines for visualization

def rot_x(theta):
  c, s = np.cos(theta), np.sin(theta)
  return np.array([
    [1, 0, 0],
    [0, c, -s],
    [0, s, c]
  ])

def rot_y(theta):
  c, s = np.cos(theta), np.sin(theta)
  return np.array([
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c]
  ])

def rot_z(theta):
  c, s = np.cos(theta), np.sin(theta)
  return np.array([
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1]
  ])

def get_joint_positions_and_orientations(bone_lengths, joint_angles):
    # Bone lengths
    spine, neck, head = bone_lengths['spine'], bone_lengths['neck'], bone_lengths['head']
    upper_arm, lower_arm = bone_lengths['upper_arm'], bone_lengths['lower_arm']
    upper_leg, lower_leg = bone_lengths['upper_leg'], bone_lengths['lower_leg']

    # Joint angles (radians)
    # [spine_yaw, spine_pitch, spine_roll, shoulder_R, elbow_R, shoulder_L, elbow_L, hip_R, knee_R, hip_L, knee_L]
    pelvis_yaw, pelvis_pitch, pelvis_roll = joint_angles[:3]
    spine_yaw, spine_pitch, spine_roll = joint_angles[3:6]
    a_sho_r, a_elb_r, a_sho_l, a_elb_l, a_hip_r, a_knee_r, a_hip_l, a_knee_l = joint_angles[6:]

    joint_positions = []
    joint_orientations = []

    # Pelvis (root)
    p_pelvis = np.array([0., 0., 0.])
    R_pelvis = np.eye(3) @ rot_y(pelvis_pitch)
    joint_positions.append(p_pelvis)
    joint_orientations.append(R_pelvis)

    # Spine top (with full orientation)
    R_spine = R_pelvis @ rot_z(spine_yaw) @ rot_y(spine_pitch) @ rot_x(spine_roll)
    p_spine = p_pelvis + R_pelvis @ np.array([0, 0, spine])
    joint_positions.append(p_spine)
    joint_orientations.append(R_spine)

    # Neck top
    p_neck = p_spine + R_spine @ np.array([0, 0, neck])
    R_neck = R_spine
    joint_positions.append(p_neck)
    joint_orientations.append(R_neck)

    # Head top
    p_head = p_neck + R_neck @ np.array([0, 0, head])
    R_head = R_neck
    joint_positions.append(p_head)
    joint_orientations.append(R_head)

    # Shoulders (Y offset, from spine top)
    shoulder_offset = 0.15
    p_sho_r = p_spine + R_spine @ np.array([0, -shoulder_offset, 0])
    R_sho_r = R_spine
    joint_positions.append(p_sho_r)
    joint_orientations.append(R_sho_r)
    p_sho_l = p_spine + R_spine @ np.array([0, shoulder_offset, 0])
    R_sho_l = R_spine
    joint_positions.append(p_sho_l)
    joint_orientations.append(R_sho_l)

    # --- RIGHT ARM CHAIN ---
    R_elbow_r = R_sho_r @ rot_z(a_sho_r)
    p_elb_r = p_sho_r + R_elbow_r @ np.array([0, -upper_arm, 0])
    joint_positions.append(p_elb_r)
    joint_orientations.append(R_elbow_r)
    R_hand_r = R_elbow_r @ rot_z(a_elb_r)
    p_hand_r = p_elb_r + R_hand_r @ np.array([0, -lower_arm, 0])
    joint_positions.append(p_hand_r)
    joint_orientations.append(R_hand_r)

    # --- LEFT ARM CHAIN ---
    R_elbow_l = R_sho_l @ rot_z(a_sho_l)
    p_elb_l = p_sho_l + R_elbow_l @ np.array([0, upper_arm, 0])
    joint_positions.append(p_elb_l)
    joint_orientations.append(R_elbow_l)
    R_hand_l = R_elbow_l @ rot_z(a_elb_l)
    p_hand_l = p_elb_l + R_hand_l @ np.array([0, lower_arm, 0])
    joint_positions.append(p_hand_l)
    joint_orientations.append(R_hand_l)

    # --- RIGHT LEG CHAIN ---
    pelvis = joint_positions[0]
    R_base = np.eye(3)
    # R_pelvis = joint_orientations[0]
    hip_offset = 0.2
    p_hip_r = pelvis + R_base @ np.array([0, -hip_offset, 0])
    R_hip_r = R_base @ rot_z(a_hip_r)
    joint_positions.append(p_hip_r)
    joint_orientations.append(R_hip_r)
    p_knee_r = p_hip_r + R_hip_r @ np.array([0, 0, -upper_leg])
    R_knee_r = R_hip_r @ rot_z(a_knee_r)
    joint_positions.append(p_knee_r)
    joint_orientations.append(R_knee_r)
    p_foot_r = p_knee_r + R_knee_r @ np.array([0, 0, -lower_leg])
    joint_positions.append(p_foot_r)
    joint_orientations.append(R_knee_r)

    # --- LEFT LEG CHAIN ---
    p_hip_l = pelvis + R_base @ np.array([0, hip_offset, 0])
    R_hip_l = R_base @ rot_z(a_hip_l)
    joint_positions.append(p_hip_l)
    joint_orientations.append(R_hip_l)
    p_knee_l = p_hip_l + R_hip_l @ np.array([0, 0, -upper_leg])
    R_knee_l = R_hip_l @ rot_z(a_knee_l)
    joint_positions.append(p_knee_l)
    joint_orientations.append(R_knee_l)
    p_foot_l = p_knee_l + R_knee_l @ np.array([0, 0, -lower_leg])
    joint_positions.append(p_foot_l)
    joint_orientations.append(R_knee_l)

    return np.vstack(joint_positions), joint_orientations

def draw_frame(ax, origin, R, length=0.1):
    """Draw a coordinate frame at the given origin with rotation R."""
    x_axis = origin + R @ np.array([length, 0, 0])
    y_axis = origin + R @ np.array([0, length, 0])
    z_axis = origin + R @ np.array([0, 0, length])
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r')
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g')
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b')

def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Example: 20 deg yaw, 10 deg pitch, 0 roll, arms T-pose, legs straight
joint_angles = [
  np.deg2rad(20), np.deg2rad(20), np.deg2rad(20),   # pelvis yaw, pitch, roll
  np.deg2rad(00), np.deg2rad(0), np.deg2rad(0),   # spine_yaw, spine_pitch, spine_roll
  np.deg2rad(70), np.deg2rad(30),                    # shoulder_R, elbow_R
  np.deg2rad(0), np.deg2rad(0),                    # shoulder_L, elbow_L
  np.deg2rad(0), np.deg2rad(0),                    # hip_R, knee_R
  np.deg2rad(0), np.deg2rad(0),                    # hip_L, knee_L
]

joint_positions, joint_orientations = get_joint_positions_and_orientations(bone_lengths, joint_angles)


fig = plt.figure(figsize=(7,9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(joint_positions[:,0], joint_positions[:,1], joint_positions[:,2], color='red', s=50)
for b in bones:
  xs,ys,zs = zip(*joint_positions[list(b)])
  ax.plot(xs, ys, zs, color='blue', linewidth=2)
    
for i, (pos, R) in enumerate(zip(joint_positions, joint_orientations)):
  draw_frame(ax, pos, R, length=axis_len)
  ax.text(pos[0], pos[1], pos[2], f'{i}', color='black', fontsize=10)

ax.set_xlabel('X (forward)')
ax.set_ylabel('Y (left)')
ax.set_zlabel('Z (up)')
ax.set_title('3D Human: Spine Orientation + Proper Orthonormal Joint Frames')
set_axes_equal(ax)

plt.show()




for i, (pos, R) in enumerate(zip(joint_positions, joint_orientations)):
    draw_frame(ax, pos, R, length=axis_len)
    ax.text(pos[0], pos[1], pos[2], f'{i}', color='black', fontsize=10)
    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left)')
    ax.set_zlabel('Z (up)')
    ax.set_title('3D Human: Spine Orientation + Proper Orthonormal Joint Frames')

    # Set equal scale for all axes
    xyz_min = joint_positions.min(axis=0)
    xyz_max = joint_positions.max(axis=0)
    max_range = (xyz_max - xyz_min).max()
    mid = (xyz_max + xyz_min) / 2
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    ax.set_box_aspect([1,1,1])
    plt.show()
