import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
  5: right elbow
  6: right hand
  7: left shoulder
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
  (4,5): right shoulder -> right elbow
  (5,6): right elbow -> right hand
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

joint_names = [
  "pelvis", "spine top", "neck top", "head top",
  "right shoulder", "right elbow", "right hand",
  "left shoulder", "left elbow", "left hand",
  "right hip", "right knee", "right foot",
  "left hip", "left knee", "left foot"
]

# -- Bone connections (from, to) --
bones_idx = [
  (0,1), (1,2), (2,3),          # pelvis->spine->neck->head
  (1,4), (4,5), (5,6),          # spine->shoulderR->elbowR->handR
  (1,7), (7,8), (8,9),          # spine->shoulderL->elbowL->handL
  (0,10), (10,11), (11,12),     # pelvis->hipR->kneeR->footR
  (0,13), (13,14), (14,15),     # pelvis->hipL->kneeL->footL
]

bone_lengths = {
  'spine': 0.5, 'neck': 0.1, 'head': 0.1,
  'upper_arm': 0.3, 'lower_arm': 0.25,
  'upper_leg': 0.4, 'lower_leg': 0.4,
  'shoulder_offset': 0.2,   # offset for shoulder joints
  'hip_offset': 0.1         # offset for hip joints
}

axis_len = 0.05  # length of axis lines for visualization
visualize_ik_iterations = False  # Set to True to visualize each IK iteration

def rot_x(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

joint_angles = [
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # pelvis yaw, pitch, roll (0, 1, 2)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # spine_yaw, spine_pitch, spine_roll (3, 4, 5)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # neck_yaw, neck_pitch, neck_roll (6, 7, 8)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # head_yaw, head_pitch, head_roll (9, 10, 11)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # shoulder_R_yaw, shoulder_R_pitch, shoulder_R_roll (12, 13, 14)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # elbow_R_yaw, elbow_R_pitch, elbow_R_roll (15, 16, 17)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # hand_R_yaw, hand_R_pitch, hand_R_roll (18, 19, 20)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # shoulder_L_yaw, shoulder_L_pitch, shoulder_L_roll (21, 22, 23)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # elbow_L_yaw, elbow_L_pitch, elbow_L_roll (24, 25, 26)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # hand_L_yaw, hand_L_pitch, hand_L_roll (27, 28, 29)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # hip_R_yaw, hip_R_pitch, hip_R_roll (30, 31, 32)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # knee_R_yaw, knee_R_pitch, knee_R_roll (33, 34, 35)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # foot_R_yaw, foot_R_pitch, foot_R_roll (36, 37, 38)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # hip_L_yaw, hip_L_pitch, hip_L_roll (39, 40, 41)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0),   # knee_L_yaw, knee_L_pitch, knee_L_roll (42, 43, 44)
  np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)    # foot_L_yaw, foot_L_pitch, foot_L_roll (45, 46, 47)
]

def right_hand_ik(
    bone_lengths, 
    joint_angles, 
    target_pos, 
    max_iters=100,
    tolerance=1e-2, 
    step_size=1
):
  """
  Simple iterative inverse kinematics for the right arm chain.
  Args:
      bone_lengths: dict of bone lengths
      joint_angles: list of 48 joint angles
      target_pos: 3D numpy array (desired hand position)
      max_iters: number of iterations
      tolerance: tolerance for position error
      step_size: step size
  Returns:
      joint_angles (modified, with new right arm angles)
  """
  joint_angles = np.array(joint_angles).copy()  # don't modify original

  # right_arm_idx = list(range(12, 21))  # indices for shoulder, elbow, hand (right)
  right_arm_idx = [3, 12, 14, 15, 18, 30, 31, 32]  # shoulder, elbow, hand (right)
  # right_arm_idx = list(range(0, 48)) # For demo, use all joints (as you had)
  angles_history = [joint_angles.copy()]
  
  for it in range(max_iters):
      # Forward kinematics: get all joint positions & orientations
      joint_positions, joint_orientations = get_joint_positions_and_orientations(bone_lengths, joint_angles)
      hand_pos = joint_positions[6]  # index 6: right hand

      error = target_pos - hand_pos
      err_norm = np.linalg.norm(error)
      if it == max_iters - 1:
          print(f"Max iterations reached ({max_iters}). Final error: {err_norm:.5f}")
      if err_norm < tolerance:
          print(f"IK converged in {it} iterations. Final error: {err_norm:.5f}")
          break

      # Numerical Jacobian: perturb each right arm joint angle and observe change in hand position
      J = np.zeros((3, len(right_arm_idx)))
      delta = 1e-2
      for j, idx in enumerate(right_arm_idx):
        orig = joint_angles[idx]
        joint_angles[idx] += delta
        joint_positions_pert, _ = get_joint_positions_and_orientations(bone_lengths, joint_angles)
        hand_pos_pert = joint_positions_pert[6]
        J[:, j] = (hand_pos_pert - hand_pos) / delta
        joint_angles[idx] = orig  # restore

      # Jacobian transpose update (gradient descent step)
      d_theta = step_size * J.T @ error
      joint_angles[right_arm_idx] += d_theta
      angles_history.append(joint_angles.copy())

  return joint_angles, angles_history

def get_joint_positions_and_orientations(bone_lengths, joint_angles):
    # Bone lengths
    spine_len, neck_len, head_len = bone_lengths['spine'], bone_lengths['neck'], bone_lengths['head']
    upper_arm_len, lower_arm_len = bone_lengths['upper_arm'], bone_lengths['lower_arm']
    upper_leg_len, lower_leg_len = bone_lengths['upper_leg'], bone_lengths['lower_leg']
    shoulder_offset = bone_lengths['shoulder_offset']
    hip_offset = bone_lengths['hip_offset']

    # Joint angles (radians)
    pelvis_yaw, pelvis_pitch, pelvis_roll = joint_angles[:3]
    spine_yaw, spine_pitch, spine_roll = joint_angles[3:6]
    neck_yaw, neck_pitch, neck_roll = joint_angles[6:9]
    head_yaw, head_pitch, head_roll = joint_angles[9:12]
    shoulder_right_yaw, shoulder_right_pitch, shoulder_right_roll = joint_angles[12:15]
    elbow_right_yaw, elbow_right_pitch, elbow_right_roll = joint_angles[15:18]
    hand_right_yaw, hand_right_pitch, hand_right_roll = joint_angles[18:21]
    shoulder_left_yaw, shoulder_left_pitch, shoulder_left_roll = joint_angles[21:24]
    elbow_left_yaw, elbow_left_pitch, elbow_left_roll = joint_angles[24:27]
    hand_left_yaw, hand_left_pitch, hand_left_roll = joint_angles[27:30]
    hip_right_yaw, hip_right_pitch, hip_right_roll = joint_angles[30:33]
    knee_right_yaw, knee_right_pitch, knee_right_roll = joint_angles[33:36]
    foot_right_yaw, foot_right_pitch, foot_right_roll = joint_angles[36:39]
    hip_left_yaw, hip_left_pitch, hip_left_roll = joint_angles[39:42]
    knee_left_yaw, knee_left_pitch, knee_left_roll = joint_angles[42:45]
    foot_left_yaw, foot_left_pitch, foot_left_roll = joint_angles[45:48]

    joint_positions = []
    joint_orientations = []

    # Pelvis (root)
    p_pelvis = np.array([0., 0., 0.])
    R_pelvis = np.eye(3) @ rot_z(pelvis_yaw) @ rot_y(pelvis_pitch) @ rot_x(pelvis_roll)
    joint_positions.append(p_pelvis)
    joint_orientations.append(R_pelvis)

    # Spine top (with full orientation)
    R_spine = R_pelvis @ rot_z(spine_yaw) @ rot_y(spine_pitch) @ rot_x(spine_roll)
    p_spine = p_pelvis + R_pelvis @ np.array([0, 0, spine_len])
    joint_positions.append(p_spine)
    joint_orientations.append(R_spine)

    # Neck top
    R_neck = R_spine @ rot_z(neck_yaw) @ rot_y(neck_pitch) @ rot_x(neck_roll)
    p_neck = p_spine + R_spine @ np.array([0, 0, neck_len])
    joint_positions.append(p_neck)
    joint_orientations.append(R_neck)

    # Head top
    R_head = R_neck @ rot_z(head_yaw) @ rot_y(head_pitch) @ rot_x(head_roll)
    p_head = p_neck + R_neck @ np.array([0, 0, head_len])
    joint_positions.append(p_head)
    joint_orientations.append(R_head)

    # --- RIGHT ARM CHAIN ---
    # Shoulder Right (Y offset, from spine top)
    R_sho_r = R_spine @ rot_z(shoulder_right_yaw) @ rot_y(shoulder_right_pitch) @ rot_x(shoulder_right_roll)
    p_sho_r = p_spine + R_spine @ np.array([0, -shoulder_offset, 0])
    joint_positions.append(p_sho_r)
    joint_orientations.append(R_sho_r)

    R_elbow_r = R_sho_r @ rot_z(elbow_right_yaw) @ rot_y(elbow_right_pitch) @ rot_x(elbow_right_roll)
    p_elb_r = p_sho_r + R_sho_r @ np.array([0, -upper_arm_len, 0]) #! wrong?
    joint_positions.append(p_elb_r)
    joint_orientations.append(R_elbow_r)

    R_hand_r = R_elbow_r @ rot_z(hand_right_yaw) @ rot_y(hand_right_pitch) @ rot_x(hand_right_roll)
    p_hand_r = p_elb_r + R_elbow_r @ np.array([0, -lower_arm_len, 0])
    joint_positions.append(p_hand_r)
    joint_orientations.append(R_hand_r)

    # --- LEFT ARM CHAIN ---
    p_sho_l = p_spine + R_spine @ np.array([0, shoulder_offset, 0])
    R_sho_l = R_spine @ rot_z(shoulder_left_yaw) @ rot_y(shoulder_left_pitch) @ rot_x(shoulder_left_roll)
    joint_positions.append(p_sho_l)
    joint_orientations.append(R_sho_l)

    R_elbow_l = R_sho_l @ rot_z(elbow_left_yaw) @ rot_y(elbow_left_pitch) @ rot_x(elbow_left_roll)
    p_elb_l = p_sho_l + R_sho_l @ np.array([0, upper_arm_len, 0])
    joint_positions.append(p_elb_l)
    joint_orientations.append(R_elbow_l)

    R_hand_l = R_elbow_l @ rot_z(hand_left_yaw) @ rot_y(hand_left_pitch) @ rot_x(hand_left_roll)
    p_hand_l = p_elb_l + R_elbow_l @ np.array([0, upper_arm_len, 0])
    joint_positions.append(p_hand_l)
    joint_orientations.append(R_hand_l)

    # --- RIGHT LEG CHAIN ---
    pelvis = joint_positions[0]
    R_base = np.eye(3)
    p_hip_r = pelvis + R_base @ np.array([0, -hip_offset, 0])
    R_hip_r = R_base @ rot_z(hip_right_yaw) @ rot_y(hip_right_pitch) @ rot_x(hip_right_roll)
    joint_positions.append(p_hip_r)
    joint_orientations.append(R_hip_r)

    p_knee_r = p_hip_r + R_hip_r @ np.array([0, 0, -upper_leg_len])
    R_knee_r = R_hip_r @ rot_z(knee_right_yaw) @ rot_y(knee_right_pitch) @ rot_x(knee_right_roll)
    joint_positions.append(p_knee_r)
    joint_orientations.append(R_knee_r)

    p_foot_r = p_knee_r + R_knee_r @ np.array([0, 0, -lower_leg_len])
    R_foot_r = R_knee_r @ rot_z(foot_right_yaw) @ rot_y(foot_right_pitch) @ rot_x(foot_right_roll)
    joint_positions.append(p_foot_r)
    joint_orientations.append(R_foot_r)

    # --- LEFT LEG CHAIN ---
    p_hip_l = pelvis + R_base @ np.array([0, hip_offset, 0])
    R_hip_l = R_base @ rot_z(hip_left_yaw) @ rot_y(hip_left_pitch) @ rot_x(hip_left_roll)
    joint_positions.append(p_hip_l)
    joint_orientations.append(R_hip_l)

    p_knee_l = p_hip_l + R_hip_l @ np.array([0, 0, -upper_leg_len])
    R_knee_l = R_hip_l @ rot_z(knee_left_yaw) @ rot_y(knee_left_pitch) @ rot_x(knee_left_roll)
    joint_positions.append(p_knee_l)
    joint_orientations.append(R_knee_l)

    p_foot_l = p_knee_l + R_knee_l @ np.array([0, 0, -lower_leg_len])
    R_foot_l = R_knee_l @ rot_z(foot_left_yaw) @ rot_y(foot_left_pitch) @ rot_x(foot_left_roll)
    joint_positions.append(p_foot_l)
    joint_orientations.append(R_foot_l)

    return np.vstack(joint_positions), joint_orientations

def draw_frame(ax, origin, R, length=0.05):
    """Draw a coordinate frame at the given origin with rotation R."""
    x_axis = origin + R @ np.array([length, 0, 0])
    y_axis = origin + R @ np.array([0, length, 0])
    z_axis = origin + R @ np.array([0, 0, length])
    axis_thickness = 3  # Change this value for thicker/thinner axes
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r', linewidth=axis_thickness)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g', linewidth=axis_thickness)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b', linewidth=axis_thickness)

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

target_hand_pos = np.array([0.5, -0.3, 0.8])  # Set your desired 3D position here
joint_angles_ik, angles_history = right_hand_ik(bone_lengths, joint_angles, target_hand_pos)

# joint_positions, joint_orientations = get_joint_positions_and_orientations(bone_lengths, joint_angles)
joint_positions, joint_orientations = get_joint_positions_and_orientations(bone_lengths, joint_angles_ik)

# Precompute all joint positions for each iteration
positions_history = []
for angles in angles_history:
    joint_positions, joint_orientations = get_joint_positions_and_orientations(bone_lengths, angles)
    positions_history.append(joint_positions.copy())
    
fig = plt.figure(figsize=(7,9))
ax = fig.add_subplot(111, projection='3d')

def animate(i):
    ax.clear()
    joint_positions = positions_history[i]
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], color='red', s=50)
    ax.scatter(target_hand_pos[0], target_hand_pos[1], target_hand_pos[2], color='green', s=150, label='Target')
    for b in bones_idx:
        xs, ys, zs = zip(*joint_positions[list(b)])
        ax.plot(xs, ys, zs, color='black', linewidth=2)
    for j, pos in enumerate(joint_positions):
        ax.text(pos[0], pos[1], pos[2], f'{j}: {joint_names[j]}', color='darkblue', fontsize=9)
    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left)')
    ax.set_zlabel('Z (up)')
    ax.set_title(f'IK Iteration {i+1}/{len(positions_history)}')
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    ax.legend(loc='upper right')

if visualize_ik_iterations:
  ani = FuncAnimation(fig, animate, frames=len(positions_history), interval=300, repeat=False)
else:
  # Plot joints
  ax.scatter(joint_positions[:,0], joint_positions[:,1], joint_positions[:,2], color='red', s=50)
  ax.scatter(target_hand_pos[0], target_hand_pos[1], target_hand_pos[2], color='green', s=150, label='Target Position')
  # Plot bones
  for b in bones_idx:
    xs,ys,zs = zip(*joint_positions[list(b)])
    ax.plot(xs, ys, zs, color='black', linewidth=2)
  # Draw joint names    
  for i, (pos, R) in enumerate(zip(joint_positions, joint_orientations)):
    draw_frame(ax, pos, R, length=0.05)
    ax.text(pos[0], pos[1], pos[2], f'{i}: {joint_names[i]}', color='darkblue', fontsize=9)

  ax.set_xlabel('X (forward)')
  ax.set_ylabel('Y (left)')
  ax.set_zlabel('Z (up)')
  ax.set_title('3D Human Skeleton Visualization')
  ax.legend(loc='upper right')
  ax.set_box_aspect([1, 1, 1])
  set_axes_equal(ax)
  plt.tight_layout()

plt.show()