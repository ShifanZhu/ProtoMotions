# ============================
# FROM-MARKERS MAIN (drop-in)
# ============================
import argparse
import os
import re
import numpy as np

# Optional: pandas makes the loader robust. If you don't want a new dep, you can
# swap this loader for a csv.reader implementation.
try:
    import pandas as pd
except Exception as _e:
    pd = None

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib as mpl
import time
from collections import defaultdict
from matplotlib import cm

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
    PELVIS, SPINE_TOP, NECK_TOP, HEAD_TOP,   # 0-3
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HAND, # 4-6
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_HAND,    # 7-9
    RIGHT_HIP, RIGHT_KNEE, RIGHT_FOOT,       # 10-12
    LEFT_HIP, LEFT_KNEE, LEFT_FOOT           # 13-15
) = range(16)

def DOF(joint, axis): return 3 * joint + axis

NECK_TOP_YAW, NECK_TOP_PITCH, NECK_TOP_ROLL = DOF(NECK_TOP, 0), DOF(NECK_TOP, 1), DOF(NECK_TOP, 2)
SPINE_TOP_YAW, SPINE_TOP_PITCH, SPINE_TOP_ROLL = DOF(SPINE_TOP, 0), DOF(SPINE_TOP, 1), DOF(SPINE_TOP, 2)
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
BONE_PAIR_TO_INDEX = {pair: i for i, pair in enumerate(BONES_IDX)}

BONE_LENGTHS = {
    'spine': 0.5, 'neck': 0.1, 'head': 0.1,
    'upper_arm': 0.3, 'lower_arm': 0.25,
    'upper_leg': 0.4, 'lower_leg': 0.4,
    'shoulder_offset': 0.2, 'hip_offset': 0.1
}

# Thick limb defaults (radii)
BONE_THICKNESS = {
    'spine': 0.10, 'neck': 0.06, 'head': 0.09,
    'upper_arm': 0.045, 'lower_arm': 0.040,
    'upper_leg': 0.070, 'lower_leg': 0.060,
    'shoulder_offset': 0.060, 'hip_offset': 0.065,
}

# ---- AMASS name -> joint index -> bone pair (parent, joint) -----------------
def _canon_ext(s: str) -> str:
    import re as _re
    return _re.sub(r'[^a-z0-9]+', '', str(s).lower())

# Kinematic parents (parent -> child pairs must match BONES_IDX topology)
PARENT = {
    SPINE_TOP: PELVIS,
    NECK_TOP:  SPINE_TOP,
    HEAD_TOP:  NECK_TOP,
    RIGHT_SHOULDER: SPINE_TOP,
    RIGHT_ELBOW:    RIGHT_SHOULDER,
    RIGHT_HAND:     RIGHT_ELBOW,
    LEFT_SHOULDER:  SPINE_TOP,
    LEFT_ELBOW:     LEFT_SHOULDER,
    LEFT_HAND:      LEFT_ELBOW,
    RIGHT_HIP:  PELVIS,
    RIGHT_KNEE: RIGHT_HIP,
    RIGHT_FOOT: RIGHT_KNEE,
    LEFT_HIP:   PELVIS,
    LEFT_KNEE:  LEFT_HIP,
    LEFT_FOOT:  LEFT_KNEE,
}

# Your provided pairs:
# [pelvis, Pelvis] [spine_top, chest] [neck_top, Neck] [head_top, Head]
# [right_shoulder, R_Shoulder] [right_elbow, R_Elbow] [right_hand, R_hand]
# [left_shoulder, L_Shoulder]  [left_elbow, L_Elbow]  [left_hand, L_hand]
# [right_hip, R_Hip] [right_knee, R_Knee] [right_foot, R_Ankle]
# [left_hip, L_Hip]  [left_knee, L_Knee] [left_foot, L_Ankle]
AMASS2JOINT = {
    "pelvis":   PELVIS,
    "chest":    SPINE_TOP,
    "neck":     NECK_TOP,
    "head":     HEAD_TOP,
    "rshoulder": RIGHT_SHOULDER,
    "relbow":    RIGHT_ELBOW,
    "rhand":     RIGHT_HAND,   # handles "R_hand" / "R_Hand" via canonicalization
    "lshoulder": LEFT_SHOULDER,
    "lelbow":    LEFT_ELBOW,
    "lhand":     LEFT_HAND,
    "rhip":      RIGHT_HIP,
    "rknee":     RIGHT_KNEE,
    "rankle":    RIGHT_FOOT,   # AMASS ankle -> our FOOT joint
    "lhip":      LEFT_HIP,
    "lknee":     LEFT_KNEE,
    "lankle":    LEFT_FOOT,
}

def build_marker_bones_from_amass_names(names: list[str]) -> list[tuple[int, int]] | None:
    """
    Returns a list of (ja, jb) per marker name using the AMASS mapping.
    For marker at joint J, we associate it to bone (parent(J), J).
    Special-case pelvis (no parent) -> use (PELVIS, SPINE_TOP).
    If any name is unknown, returns None (so caller can fall back to auto-assign).
    """
    mb: list[tuple[int, int] | None] = []
    unknown = []
    for nm in names:
        j = AMASS2JOINT.get(_canon_ext(nm))
        if j is None:
            unknown.append(nm)
            mb.append(None)
            continue
        if j in PARENT:
            a = PARENT[j]; b = j
        else:
            # only pelvis has no parent; tie it to its child bone
            a, b = PELVIS, SPINE_TOP
        mb.append((a, b))

    if any(p is None for p in mb):
        print("[amass-map] WARNING: unknown marker names (falling back to auto-assign):",
              ", ".join(map(str, unknown)))
        return None
    return [p for p in mb]  # type: ignore


def default_bone_radii():
    """Return radii aligned with BONES_IDX order."""
    radii = []
    for (ja, jb) in BONES_IDX:
        if   (ja, jb) == (PELVIS, SPINE_TOP):          r = BONE_THICKNESS['spine']
        elif (ja, jb) == (SPINE_TOP, NECK_TOP):        r = BONE_THICKNESS['neck']
        elif (ja, jb) == (NECK_TOP, HEAD_TOP):         r = BONE_THICKNESS['head']
        elif (ja, jb) in [(SPINE_TOP, RIGHT_SHOULDER), (SPINE_TOP, LEFT_SHOULDER)]:
            r = BONE_THICKNESS['shoulder_offset']
        elif (ja, jb) in [(RIGHT_SHOULDER, RIGHT_ELBOW), (LEFT_SHOULDER, LEFT_ELBOW)]:
            r = BONE_THICKNESS['upper_arm']
        elif (ja, jb) in [(RIGHT_ELBOW, RIGHT_HAND), (LEFT_ELBOW, LEFT_HAND)]:
            r = BONE_THICKNESS['lower_arm']
        elif (ja, jb) in [(PELVIS, RIGHT_HIP), (PELVIS, LEFT_HIP)]:
            r = BONE_THICKNESS['hip_offset']
        elif (ja, jb) in [(RIGHT_HIP, RIGHT_KNEE), (LEFT_HIP, LEFT_KNEE)]:
            r = BONE_THICKNESS['upper_leg']
        elif (ja, jb) in [(RIGHT_KNEE, RIGHT_FOOT), (LEFT_KNEE, LEFT_FOOT)]:
            r = BONE_THICKNESS['lower_leg']
        else:
            r = 0.05
        radii.append(r)
    return np.asarray(radii, dtype=float)

# Keys of bone lengths to optimize
BONE_LENGTH_KEYS_TO_OPTIMIZE = ['upper_arm', 'lower_arm', 'upper_leg', 'lower_leg', 'neck']

def update_bone_lengths_from_vec(bone_lengths, vec):
    for k, v in zip(BONE_LENGTH_KEYS_TO_OPTIMIZE, vec):
        bone_lengths[k] = v
    return bone_lengths

def get_default_joint_angles():
    return np.zeros(48)

def _deg(a):  # degrees -> radians
    return np.deg2rad(a)

def get_default_joint_limits():
    lower = -np.pi * np.ones(48, dtype=float)
    upper =  np.pi * np.ones(48, dtype=float)

    def set_limits(joint, yaw_range, pitch_range, roll_range):
        y0, y1 = _deg(yaw_range[0]), _deg(yaw_range[1])
        p0, p1 = _deg(pitch_range[0]), _deg(pitch_range[1])
        r0, r1 = _deg(roll_range[0]), _deg(roll_range[1])
        i = 3 * joint
        lower[i + 0], upper[i + 0] = y0, y1
        lower[i + 1], upper[i + 1] = p0, p1
        lower[i + 2], upper[i + 2] = r0, r1

    set_limits(PELVIS,       (-180, 180), (-90, 90),  (-90, 90))
    set_limits(SPINE_TOP,    (-60,  60),  (-45, 45),  (-45, 45))
    set_limits(NECK_TOP,     (-80,  80),  (-60, 60),  (-60, 60))
    set_limits(HEAD_TOP,     (-80,  80),  (-60, 60),  (-60, 60))
    set_limits(RIGHT_SHOULDER, (-150, 150), (-150, 150), (-100, 100))
    set_limits(LEFT_SHOULDER,  (-150, 150), (-150, 150), (-100, 100))
    set_limits(RIGHT_ELBOW,  (-45, 45), (0, 150), (-45, 45))
    set_limits(LEFT_ELBOW,   (-45, 45), (0, 150), (-45, 45))
    set_limits(RIGHT_HAND,   (-90, 90), (-90, 90), (-90, 90))
    set_limits(LEFT_HAND,    (-90, 90), (-90, 90), (-90, 90))
    set_limits(RIGHT_HIP,    (-70, 70), (-120, 120), (-50, 50))
    set_limits(LEFT_HIP,     (-70, 70), (-120, 120), (-50, 50))
    set_limits(RIGHT_KNEE,   (-30, 30), (0, 150), (-30, 30))
    set_limits(LEFT_KNEE,    (-30, 30), (0, 150), (-30, 30))
    set_limits(RIGHT_FOOT,   (-45, 45), (-45, 45), (-30, 30))
    set_limits(LEFT_FOOT,    (-45, 45), (-45, 45), (-30, 30))
    return lower, upper

def rot_x(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(theta): c,s = np.cos(theta), np.sin(theta); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def get_joint_positions_and_orientations(bone_lengths, joint_angles, root_pos=np.zeros(3)):
    """
    FK with an explicit root (pelvis) translation 'root_pos' in world coordinates.
    """
    spine_len, neck_len, head_len = bone_lengths['spine'], bone_lengths['neck'], bone_lengths['head']
    upper_arm_len, lower_arm_len = bone_lengths['upper_arm'], bone_lengths['lower_arm']
    upper_leg_len, lower_leg_len = bone_lengths['upper_leg'], bone_lengths['lower_leg']
    shoulder_offset, hip_offset = bone_lengths['shoulder_offset'], bone_lengths['hip_offset']

    def ang(idx): return joint_angles[3*idx:3*idx+3]
    joint_positions, joint_orientations = [], []

    p = np.array(root_pos, dtype=float)
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
    p_hand_l = p_elb_l + R_elb_l @ np.array([0, lower_arm_len, 0])
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

# ============== #
# Line/segment residuals (legacy)
# ============== #
def residual_point_to_line(marker, pa, pb):
    v = pb - pa
    L = np.linalg.norm(v)
    if L < 1e-12:
        return marker - pa
    u = v / L
    P = np.eye(3) - np.outer(u, u)
    return P @ (marker - pa)

def residual_point_to_segment(marker, pa, pb):
    v = pb - pa
    L2 = v @ v
    if L2 < 1e-12:
        return marker - pa
    t = (marker - pa) @ v / L2
    t = np.clip(t, 0.0, 1.0)
    closest = pa + t * v
    return marker - closest

# ====================== #
# Thick geometry helpers #
# ====================== #
def _perp_basis(u):
    u = u / (np.linalg.norm(u) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n1 = np.cross(u, tmp)
    n1 /= (np.linalg.norm(n1) + 1e-12)
    n2 = np.cross(u, n1)
    n2 /= (np.linalg.norm(n2) + 1e-12)
    return n1, n2

def _closest_point_on_segment_pointwise(p, a, b):
    v = b - a
    L2 = v @ v
    if L2 < 1e-12:
        return a
    t = (p - a) @ v / L2
    t = np.clip(t, 0.0, 1.0)
    return a + t * v

def _closest_point_on_segment(marker, pa, pb):
    return _closest_point_on_segment_pointwise(marker, pa, pb)

def _closest_point_on_line(marker, pa, pb):
    v = pb - pa
    L2 = v @ v
    if L2 < 1e-12:
        return pa
    t = (marker - pa) @ v / L2
    return pa + t * v

def closest_point_on_capsule_surface(p, a, b, R):
    q = _closest_point_on_segment_pointwise(p, a, b)
    d = p - q
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        v = b - a
        e = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n = np.cross(v, e); n /= (np.linalg.norm(n) + 1e-12)
        return q + R * n
    return q + (R / dn) * d

def unsigned_distance_to_capsule(p, a, b, R):
    q = _closest_point_on_segment_pointwise(p, a, b)
    d = np.linalg.norm(p - q)
    return max(0.0, d - R)

def closest_point_on_capped_cylinder_surface(p, a, b, R):
    v = b - a
    L = np.linalg.norm(v)
    if L < 1e-12:
        d = p - a
        dn = np.linalg.norm(d)
        if dn < 1e-12: return a + np.array([R,0,0])
        return a + (R/dn)*d
    u = v / L
    w = p - a
    t = w @ u
    r_perp = w - t * u
    r = np.linalg.norm(r_perp)
    if 0.0 <= t <= L:
        if r > 1e-12:
            return (a + t * u) + (R / r) * r_perp
        else:
            n1, _ = _perp_basis(u)
            return (a + t * u) + R * n1
    elif t < 0.0:
        p_plane = p - t * u
        r_vec = p_plane - a
        r = np.linalg.norm(r_vec)
        if r <= R or r < 1e-12:
            return p_plane
        else:
            return a + (R / r) * r_vec
    else:
        w2 = p - b
        t2 = w2 @ u
        p_plane = p - t2 * u
        r_vec = p_plane - b
        r = np.linalg.norm(r_vec)
        if r <= R or r < 1e-12:
            return p_plane
        else:
            return b + (R / r) * r_vec

def unsigned_distance_to_capped_cylinder(p, a, b, R):
    s = closest_point_on_capped_cylinder_surface(p, a, b, R)
    return np.linalg.norm(p - s)

def residual_point_to_capsule(marker, pa, pb, radius):
    # if unsigned_distance_to_capsule(marker, pa, pb, radius) <= 0.0:
    #     return np.zeros(3)
    s = closest_point_on_capsule_surface(marker, pa, pb, radius)
    return marker - s

def residual_point_to_cylinder(marker, pa, pb, radius):
    # if unsigned_distance_to_capped_cylinder(marker, pa, pb, radius) <= 0.0:
    #     return np.zeros(3)
    s = closest_point_on_capped_cylinder_surface(marker, pa, pb, radius)
    return marker - s

# ---- Vectorized geometry (speed) --------------------------------------------

def _perp_basis_vec(u):
    eps = 1e-12
    mask = np.abs(u[:, 0]) < 0.9
    tmp = np.empty_like(u)
    tmp[mask]  = np.array([1.0, 0.0, 0.0])
    tmp[~mask] = np.array([0.0, 1.0, 0.0])
    n1 = np.cross(u, tmp)
    n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + eps)
    n2 = np.cross(u, n1)
    n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + eps)
    return n1, n2

def _closest_on_segment_vec(m, a, b):
    eps = 1e-12
    v   = b - a
    L2  = np.sum(v*v, axis=1)
    t   = np.einsum('ij,ij->i', (m - a), v) / (L2 + eps)
    t   = np.clip(t, 0.0, 1.0)
    q   = a + t[:, None] * v
    return q, t, v

def _closest_on_line_vec(m, a, b):
    eps = 1e-12
    v   = b - a
    L2  = np.sum(v*v, axis=1)
    t   = np.einsum('ij,ij->i', (m - a), v) / (L2 + eps)
    q   = a + t[:, None] * v
    return q, t, v

def _capsule_surface_vec(m, a, b, R):
    eps = 1e-12
    q, t, v = _closest_on_segment_vec(m, a, b)
    d  = m - q
    dn = np.linalg.norm(d, axis=1)
    u  = v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)
    n1, _ = _perp_basis_vec(u)
    dir_fallback = n1
    scale = (R / (dn + eps))[:, None]
    cp = q + scale * d
    mask = dn < 1e-9
    if np.any(mask):
        cp[mask] = q[mask] + R[mask, None] * dir_fallback[mask]
    return cp

def _capped_cylinder_surface_vec(m, a, b, R):
    eps = 1e-12
    v   = b - a
    L   = np.linalg.norm(v, axis=1)
    u   = v / (L[:, None] + eps)
    w   = m - a
    t   = np.einsum('ij,ij->i', w, u)
    mid = (t >= 0.0) & (t <= L + eps)
    r_perp = w - t[:, None] * u
    r = np.linalg.norm(r_perp, axis=1) + eps
    cp_mid = (a + t[:, None] * u) + (R / r)[:, None] * r_perp
    # start cap
    before = t < 0.0
    p_plane_s = m - t[:, None] * u
    r_vec_s = p_plane_s - a
    r_s = np.linalg.norm(r_vec_s, axis=1) + eps
    cp_s = np.where(((r_s <= R + 1e-12)[:, None]), p_plane_s, a + (R / r_s)[:, None] * r_vec_s)
    # end cap
    w2 = m - b
    t2 = np.einsum('ij,ij->i', w2, u)
    after = t > L
    p_plane_e = m - t2[:, None] * u
    r_vec_e = p_plane_e - b
    r_e = np.linalg.norm(r_vec_e, axis=1) + eps
    cp_e = np.where(((r_e <= R + 1e-12)[:, None]), p_plane_e, b + (R / r_e)[:, None] * r_vec_e)
    # stitch
    cp = np.empty_like(m)
    cp[mid]    = cp_mid[mid]
    cp[before] = cp_s[before]
    cp[after]  = cp_e[after]
    return cp

def build_residual_stack_hard_geom_vec(jp, markers, marker_bones, *, geom="segment", bone_radii=None):
    """
    Vectorized residual builder for 'hard' mode on a subset (batch) of markers.
    Returns shape (3*K_batch,)
    """
    K = len(markers)
    if K == 0:
        return np.zeros((0,), dtype=float)
    ja = np.fromiter((p[0] for p in marker_bones), dtype=int, count=K)
    jb = np.fromiter((p[1] for p in marker_bones), dtype=int, count=K)
    a  = jp[ja]                  # (K,3)
    b  = jp[jb]                  # (K,3)
    m  = np.asarray(markers)     # (K,3)

    if geom == "segment":
        q, _, _ = _closest_on_segment_vec(m, a, b)
        r = m - q
    elif geom == "line":
        q, _, _ = _closest_on_line_vec(m, a, b)
        r = m - q
    elif geom in ("capsule", "cylinder"):
        if bone_radii is None:
            q, _, _ = _closest_on_segment_vec(m, a, b)
            r = m - q
        else:
            idx = np.fromiter((BONE_PAIR_TO_INDEX[p] for p in marker_bones), dtype=int, count=K)
            R = bone_radii[idx]
            if geom == "capsule":
                s = _capsule_surface_vec(m, a, b, R)
            else:
                s = _capped_cylinder_surface_vec(m, a, b, R)
            r = m - s
    else:
        raise ValueError("unknown geom")
    return r.reshape(3 * K)

# ---- Scale & groups ----------------------------------------------------------
def compute_skeleton_scale(bone_lengths):
    return (
        bone_lengths['upper_leg'] + bone_lengths['lower_leg'] +
        bone_lengths['spine'] + bone_lengths['neck'] + bone_lengths['head']
    )

def bone_groups(bones_idx=BONES_IDX):
    g = defaultdict(set)
    for bi, (ja, jb) in enumerate(bones_idx):
        if (ja, jb) in [(PELVIS, SPINE_TOP), (SPINE_TOP, NECK_TOP)]: g['torso'].add(bi)
        if (ja, jb) == (NECK_TOP, HEAD_TOP): g['head'].add(bi)
        if ja == SPINE_TOP and jb in (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HAND): g['right_arm'].add(bi)
        if ja == SPINE_TOP and jb in (LEFT_SHOULDER, LEFT_ELBOW, LEFT_HAND): g['left_arm'].add(bi)
        if ja in (RIGHT_SHOULDER, RIGHT_ELBOW): g['right_arm'].add(bi)
        if ja in (LEFT_SHOULDER, LEFT_ELBOW):   g['left_arm'].add(bi)
        if ja == PELVIS and jb in (RIGHT_HIP, LEFT_HIP):
            g['right_leg' if jb == RIGHT_HIP else 'left_leg'].add(bi)
        if ja in (RIGHT_HIP, RIGHT_KNEE): g['right_leg'].add(bi)
        if ja in (LEFT_HIP, LEFT_KNEE):   g['left_leg'].add(bi)
    g['arms']  = g['left_arm']  | g['right_arm']
    g['legs']  = g['left_leg']  | g['right_leg']
    g['upper_body'] = g['torso'] | g['head'] | g['arms']
    g['lower_body'] = g['legs']
    g['all'] = set(range(len(bones_idx)))
    return g

def expand_semantic_priors(semantic_priors, bones_idx=BONES_IDX):
    if not semantic_priors:
        return {}
    groups = bone_groups(bones_idx)
    out = {}
    for mi, allowed in semantic_priors.items():
        s = set()
        items = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        for item in items:
            if isinstance(item, str) and item in groups:
                s |= groups[item]
            elif isinstance(item, int):
                s.add(item)
            elif isinstance(item, tuple) and len(item) == 2:
                try:
                    s.add(bones_idx.index(item))
                except ValueError:
                    pass
        out[int(mi)] = s
    return out

# ---- Distances used in assignment -------------------------------------------
def _point_to_segment_distance(marker, pa, pb):
    return np.linalg.norm(marker - _closest_point_on_segment_pointwise(marker, pa, pb))

def _point_to_line_distance(marker, pa, pb):
    v = pb - pa
    L = np.linalg.norm(v)
    if L < 1e-12:
        return np.linalg.norm(marker - pa)
    u = v / L
    P = np.eye(3) - np.outer(u, u)
    return np.linalg.norm(P @ (marker - pa))

# ---- Robust correspondence core ---------------------------------------------
def robust_assign_markers(
    markers, joint_positions, bones_idx=BONES_IDX, *,
    use_segment=True,
    prev_state=None,
    bone_lengths=BONE_LENGTHS,
    # toggles & params:
    topk=1, soft_sigma_factor=0.1,
    distance_gate_abs=None, distance_gate_factor=1.0, enable_gate=True,
    hysteresis_margin=0.10, enable_hysteresis=True,
    temporal_smoothing=0.0, enable_temporal_smoothing=False,
    semantic_priors=None,
    # geometry:
    geom="segment",                  # "line" | "segment" | "capsule" | "cylinder"
    bone_radii=None                  # np.array [len(BONES_IDX)] if capsule/cylinder
):
    n_bones = len(bones_idx)
    scale = compute_skeleton_scale(bone_lengths)
    sigma = max(1e-6, soft_sigma_factor * scale)
    gate = (distance_gate_abs if (enable_gate and distance_gate_abs is not None)
            else (distance_gate_factor * scale if enable_gate else np.inf))
    allowed_by_marker = expand_semantic_priors(semantic_priors, bones_idx)

    prev_hard = prev_state.get('hard') if prev_state else None
    prev_wts  = prev_state.get('weights') if prev_state else None

    cands, weights, hard_pairs = [], [], []

    for mi, m in enumerate(np.asarray(markers)):
        allowed = allowed_by_marker.get(mi, None)
        dists, idxs = [], []
        for bi, (ja, jb) in enumerate(bones_idx):
            if (allowed is not None) and (bi not in allowed):
                continue
            pa, pb = joint_positions[ja], joint_positions[jb]

            if geom == "segment":
                d = _point_to_segment_distance(m, pa, pb)
            elif geom == "line":
                d = _point_to_line_distance(m, pa, pb)
            elif geom == "capsule":
                R = 0.0 if bone_radii is None else float(bone_radii[bi])
                d = unsigned_distance_to_capsule(m, pa, pb, R)
            elif geom == "cylinder":
                R = 0.0 if bone_radii is None else float(bone_radii[bi])
                d = unsigned_distance_to_capped_cylinder(m, pa, pb, R)
            else:
                raise ValueError("unknown geom")

            if not enable_gate or d <= gate:
                dists.append(d); idxs.append(bi)

        if not idxs:
            best_d = np.inf; best_bi = None
            for bi, (ja, jb) in enumerate(bones_idx):
                pa, pb = joint_positions[ja], joint_positions[jb]
                if geom == "segment":
                    d = _point_to_segment_distance(m, pa, pb)
                elif geom == "line":
                    d = _point_to_line_distance(m, pa, pb)
                elif geom == "capsule":
                    R = 0.0 if bone_radii is None else float(bone_radii[bi])
                    d = unsigned_distance_to_capsule(m, pa, pb, R)
                else:
                    R = 0.0 if bone_radii is None else float(bone_radii[bi])
                    d = unsigned_distance_to_capped_cylinder(m, pa, pb, R)
                if d < best_d: best_d, best_bi = d, bi
            idxs = [best_bi]; dists = [best_d]

        order = np.argsort(dists)
        idxs  = [idxs[i] for i in order][:max(1, topk)]
        dists = [dists[i] for i in order][:max(1, topk)]

        if enable_hysteresis and prev_hard is not None and mi < len(prev_hard) and prev_hard[mi] is not None:
            prev_bi = prev_hard[mi]
            if prev_bi not in idxs:
                ja, jb = bones_idx[prev_bi]
                pa, pb = joint_positions[ja], joint_positions[jb]
                if geom == "segment":
                    d_prev = _point_to_segment_distance(m, pa, pb)
                elif geom == "line":
                    d_prev = _point_to_line_distance(m, pa, pb)
                elif geom == "capsule":
                    R = 0.0 if bone_radii is None else float(bone_radii[prev_bi])
                    d_prev = unsigned_distance_to_capsule(m, pa, pb, R)
                else:
                    R = 0.0 if bone_radii is None else float(bone_radii[prev_bi])
                    d_prev = unsigned_distance_to_capped_cylinder(m, pa, pb, R)
                d_best = dists[0]
                if d_prev <= (1.0 + hysteresis_margin) * d_best:
                    idxs = [prev_bi]; dists = [d_prev]

        ws = np.exp(-0.5 * (np.array(dists) / sigma) ** 2)
        if ws.sum() <= 1e-12: ws = np.ones_like(ws)
        ws = ws / ws.sum()

        if enable_temporal_smoothing and prev_wts is not None and mi < len(prev_wts) and prev_wts[mi]:
            prev_dict = prev_wts[mi]
            merged = defaultdict(float)
            for bi, wv in zip(idxs, ws):
                merged[bi] += (1.0 - temporal_smoothing) * wv
            for bi, wv in prev_dict.items():
                merged[bi] += temporal_smoothing * wv
            items = sorted(merged.items(), key=lambda x: -x[1])[:max(1, topk)]
            idxs = [bi for bi, _ in items]
            ws   = np.array([w for _, w in items], dtype=float)
            ws = ws / ws.sum()

        hard_bi = idxs[int(np.argmax(ws))]
        hard_pairs.append(bones_idx[hard_bi])
        cands.append(idxs)
        weights.append(ws)

    next_state = {'hard': [], 'weights': []}
    for idxs, ws in zip(cands, weights):
        best_idx = idxs[int(np.argmax(ws))]
        next_state['hard'].append(best_idx)
        next_state['weights'].append({bi: float(w) for bi, w in zip(idxs, ws)})

    return {
        'mode': 'soft' if topk > 1 else 'hard',
        'hard': hard_pairs,
        'cands': cands,
        'weights': weights,
        'state': next_state
    }

# ---- Residual builders (geometry-aware) -------------------------------------
def build_residual_stack_hard_geom(jp, markers, marker_bones, *, geom="segment", bone_radii=None):
    K = len(markers)
    res = np.zeros((3 * K,), dtype=float)
    for k, (ja, jb) in enumerate(marker_bones):
        pa, pb = jp[ja], jp[jb]
        if geom == "segment":
            r = residual_point_to_segment(markers[k], pa, pb)
        elif geom == "line":
            r = residual_point_to_line(markers[k], pa, pb)
        elif geom == "capsule":
            bi = BONE_PAIR_TO_INDEX[(ja, jb)]
            R = 0.0 if bone_radii is None else float(bone_radii[bi])
            r = residual_point_to_capsule(markers[k], pa, pb, R)
        elif geom == "cylinder":
            bi = BONE_PAIR_TO_INDEX[(ja, jb)]
            R = 0.0 if bone_radii is None else float(bone_radii[bi])
            r = residual_point_to_cylinder(markers[k], pa, pb, R)
        else:
            raise ValueError("unknown geom")
        res[3*k:3*k+3] = r
    return res

def build_residual_stack_soft_geom(jp, markers, bones_idx, cands, weights, *, geom="segment", bone_radii=None):
    K = len(markers)
    res = np.zeros((3 * K,), dtype=float)
    for k in range(K):
        m = markers[k]
        closest_sum = np.zeros(3)
        for bi, w in zip(cands[k], weights[k]):
            ja, jb = bones_idx[bi]
            pa, pb = jp[ja], jp[jb]
            if geom == "segment":
                cp = _closest_point_on_segment(m, pa, pb)
            elif geom == "line":
                cp = _closest_point_on_line(m, pa, pb)
            elif geom == "capsule":
                R = 0.0 if bone_radii is None else float(bone_radii[bi])
                cp = closest_point_on_capsule_surface(m, pa, pb, R)
            elif geom == "cylinder":
                R = 0.0 if bone_radii is None else float(bone_radii[bi])
                cp = closest_point_on_capped_cylinder_surface(m, pa, pb, R)
            closest_sum += w * cp
        res[3*k:3*k+3] = m - closest_sum
    return res

# ============================== #
# Marker template + rendering    #
# ============================== #
# A template is a marker layout in bone-local coordinates. Instead of storing markers in world space (which changes 
# every frame), the template stores where each marker lives relative to a bone:
# bone index (or the pair (ja, jb)): which bone the marker belongs to
# t: the normalized position along the bone segment (t=0 at joint ja, t=1 at joint jb)
# φ (phi): the angle around the bone’s cross-section (used for thick geometry like cylinders/capsules)
# (optionally) geom: "segment" | "cylinder" | "capsule"
# (optionally) radius: the bone’s radius (if cylinder/capsule)
# (optionally) a small tangent jitter seed/std for realism
def make_marker_template(bones_idx=BONES_IDX, *, markers_per_bone=8, geom="cylinder", seed=0):
    """
    Create a per-bone template of (t, phi) samples to get consistent surface points each frame.
    For 'segment' geometry, phi is ignored.
    """
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.05, 0.95, markers_per_bone)  # avoid bone endpoints
    template = []
    for bi, _ in enumerate(bones_idx):
        # add a random rotational offset so rings don't align across bones
        rot_off = rng.random() * 2.0 * np.pi
        bone_entries = []
        for k, t in enumerate(ts):
            phi = (2.0 * np.pi * k / markers_per_bone) + rot_off
            bone_entries.append((float(t), float(phi)))
        template.append(bone_entries)
    return {"geom": geom, "entries": template, "markers_per_bone": markers_per_bone}

def render_markers_from_template(jp, template, *, bone_radii=None, jitter_tangent_std=3.0, seed=0):
    """
    Given joint positions for one frame, render the world-space marker positions.
    Returns (markers[K,3], marker_bones[list of (ja,jb)]) with consistent order across frames.
    """
    rng = np.random.default_rng(seed)
    geom = template["geom"]
    entries = template["entries"]
    markers = []
    marker_bones = []

    for bi, (ja, jb) in enumerate(BONES_IDX):
        a, b = jp[ja], jp[jb]
        v = b - a
        L = np.linalg.norm(v)
        if L < 1e-9:
            continue
        u = v / L # compute bone direction
        n1, n2 = _perp_basis(u) # build an orthonormal cross-section basis (n1, n2) perpendicular to u
        R = 0.0 if (bone_radii is None) else float(bone_radii[bi])
        for (t, phi) in entries[bi]:
            c = a + t * v
            if geom == "segment":
                p = c
                if jitter_tangent_std > 0.0:
                    p = p + (rng.normal(0.0, jitter_tangent_std) * n1 +
                             rng.normal(0.0, jitter_tangent_std) * n2)
            elif geom in ("cylinder", "capsule"):
                ring_dir = np.cos(phi) * n1 + np.sin(phi) * n2
                p = c + R * ring_dir
                if jitter_tangent_std > 0.0:
                    p = p + (rng.normal(0.0, jitter_tangent_std) * n1 +
                             rng.normal(0.0, jitter_tangent_std) * n2)
                    if geom == "cylinder":
                        p = closest_point_on_capped_cylinder_surface(p, a, b, R)
                    else:
                        p = closest_point_on_capsule_surface(p, a, b, R)
            else:
                raise ValueError("geom must be 'segment', 'cylinder', or 'capsule'")
            markers.append(p)
            marker_bones.append((ja, jb))
    return np.asarray(markers), marker_bones

# ============ #
# Visualization
# ============ #
def draw_frame(ax, origin, R, length=0.05):
    x_axis = origin + R @ np.array([length, 0, 0])
    y_axis = origin + R @ np.array([0, length, 0])
    z_axis = origin + R @ np.array([0, 0, length])
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r', linewidth=2)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g', linewidth=2)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b', linewidth=2)

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

def draw_cylinder(ax, a, b, R, color=(0.75,0.75,0.8), alpha=0.25, n_theta=18, n_len=6):
    v = b - a
    L = np.linalg.norm(v)
    if L < 1e-9:
        return
    u = v / L
    n1, n2 = _perp_basis(u)
    t_vals = np.linspace(0.0, L, n_len)
    th = np.linspace(0, 2*np.pi, n_theta)
    ct, st = np.cos(th), np.sin(th)
    X = []; Y = []; Z = []
    for t in t_vals:
        c = a + t * u
        ring = np.outer(ct, n1) + np.outer(st, n2)
        pts = c + R * ring
        X.append(pts[:,0]); Y.append(pts[:,1]); Z.append(pts[:,2])
    X = np.vstack(X); Y = np.vstack(Y); Z = np.vstack(Z)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, color=color, alpha=alpha, shade=True)

def draw_skeleton_wire(ax, joint_positions, color='k', lw=2, alpha=1.0):
    for b in BONES_IDX:
        xs, ys, zs = zip(*joint_positions[list(b)])
        ax.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha)

def plot_skeleton(
    ax, joint_positions, joint_orientations,
    markers=None, marker_bones=None, show_axes=False,
    title='', draw_solids=False, bone_radii=None, clear=True,
    joint_color='red', wire_color='black', wire_alpha=1.0
):
    if clear: ax.clear()
    # joints
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], color=joint_color, s=35, alpha=wire_alpha)
    # wire
    draw_skeleton_wire(ax, joint_positions, color=wire_color, lw=2, alpha=wire_alpha)
    # solids
    if draw_solids and bone_radii is not None:
        for bi, (ja, jb) in enumerate(BONES_IDX):
            R = float(bone_radii[bi])
            a, bpt = joint_positions[ja], joint_positions[jb]
            draw_cylinder(ax, a, bpt, R, alpha=0.2)
    # markers (optional)
    if markers is not None and len(markers) > 0:
        ax.scatter(markers[:, 0], markers[:, 1], markers[:, 2], marker='x', s=35, label='markers', color='C1')
    # axes frame (optional)
    if show_axes:
        for pos, R in zip(joint_positions, joint_orientations):
            draw_frame(ax, pos, R, length=0.05)
    ax.set_xlabel('X (forward)'); ax.set_ylabel('Y (left)'); ax.set_zlabel('Z (up)')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)

# ============================== #
# Optimizer (LM + variants)      #
#  + root translation support    #
# ============================== #
def lm_fit_markers_to_bones(
    bone_lengths,
    joint_angles,
    markers,
    marker_bones=None,
    opt_joint_indices_list=None,
    use_segment=True,
    optimize_bones=False,
    optimize_root=True,
    root_init=np.zeros(3),
    max_iters=100,
    tolerance=1e-3,
    angle_delta=1e-3,
    length_delta=1e-3,
    root_delta=1e-3,
    lm_lambda0=1e-2,
    lm_lambda_factor=2.0,
    lm_lambda_min=1e-6,
    lm_lambda_max=1e+2,
    angle_step_clip=np.deg2rad(12.0),
    length_step_clip=0.02,              # meters; set None to disable
    root_step_clip=0.05,
    bone_clip=(0.05, 2.0),
    angle_reg=1.0,
    bone_reg=5.0,
    root_reg=0.5,
    marker_weights=None,
    joint_limits=None,
    verbose=False,
    # robust assignment controls
    auto_assign_bones=False,
    assign_topk=1,
    assign_soft_sigma_factor=0.1,
    assign_distance_gate_abs=None,
    assign_distance_gate_factor=1.0,
    assign_enable_gate=True,
    assign_enable_hysteresis=True,
    assign_hysteresis_margin=0.10,
    assign_enable_temporal_smoothing=False,
    assign_temporal_smoothing=0.0,
    assign_semantic_priors=None,
    # step strategies
    strategy="lm",                      # "lm" | "lm+linesearch" | "lm+dogleg"
    line_search_scales=(1.0, 0.5, 0.25, 0.125),
    allow_trial_reassign=False,
    tr_radius0=0.15,
    tr_radius_max=1.0,
    tr_eta=0.10,
    tr_expand=2.5,
    tr_shrink=0.25,
    # geometry controls
    geom="segment",                  # "line" | "segment" | "capsule" | "cylinder"
    bone_radii=None,
    # speed knobs
    marker_batch_size=None,        # None = use all markers each iter
    reassign_every=3,              # recompute correspondences every N iters
    fast_vectorized=True,          # vectorized hard residuals
    rng_seed=0
):
    """
    Returns: (theta, bone_lengths_final, root_final, angles_history, bone_length_history, root_history)
    """
    theta = np.array(joint_angles, dtype=float).copy()
    root  = np.array(root_init, dtype=float).copy()
    bl_keys = BONE_LENGTH_KEYS_TO_OPTIMIZE if optimize_bones else []
    bl_vec = np.array([bone_lengths[k] for k in bl_keys], dtype=float)

    n_joints = theta.size
    if opt_joint_indices_list is None:
        opt_joint_indices_list = [list(range(n_joints))]
    active_angle_idx = sorted(set(i for idxs in opt_joint_indices_list for i in idxs)) or list(range(n_joints))
    n_active = len(active_angle_idx)

    K = len(markers)
    markers = np.asarray(markers, dtype=float).reshape(K, 3)

    if marker_weights is None:
        marker_weights = np.ones(K, dtype=float)
    w = np.asarray(marker_weights, dtype=float).clip(min=0.0)
    w_sqrt_full = np.sqrt(w)

    if joint_limits is None:
        lower_lim = -np.inf * np.ones(n_joints)
        upper_lim =  np.inf * np.ones(n_joints)
    else:
        lower_lim, upper_lim = joint_limits

    angles_history = [theta.copy()]
    bone_length_history = [bl_vec.copy()]
    root_history = [root.copy()]

    lm_lambda = float(lm_lambda0)
    tr_radius = float(tr_radius0)

    rng = np.random.default_rng(rng_seed)
    def _pick_indices():
        if (marker_batch_size is None) or (marker_batch_size >= K):
            return np.arange(K, dtype=int)
        return rng.choice(K, size=marker_batch_size, replace=False)

    def fk_positions(curr_theta, curr_bl_vec, curr_root):
        bl_all = bone_lengths.copy()
        for k, v in zip(bl_keys, curr_bl_vec):
            bl_all[k] = v
        jp, _ = get_joint_positions_and_orientations(bl_all, curr_theta, root_pos=curr_root)
        return jp, bl_all

    def _split(delta):
        off = n_active
        dth = delta[:off]
        off2 = off + bl_vec.size
        dbl = delta[off:off2]
        droot = delta[off2:off2+(3 if optimize_root else 0)]
        return dth, dbl, droot

    def _clip_steps(dth, dbl, droot):
        if angle_step_clip is not None and dth.size:
            mx = np.max(np.abs(dth))
            if mx > angle_step_clip:
                dth *= angle_step_clip / (mx + 1e-12)
        if length_step_clip is not None and dbl.size:
            mx = np.max(np.abs(dbl))
            if mx > length_step_clip:
                dbl *= length_step_clip / (mx + 1e-12)
        if optimize_root and root_step_clip is not None and droot.size:
            mx = np.max(np.abs(droot))
            if mx > root_step_clip:
                droot *= root_step_clip / (mx + 1e-12)
        return dth, dbl, droot

    def _propose(theta_base, bl_vec_base, root_base, dth, dbl, droot):
        th_new = theta_base.copy()
        th_new[active_angle_idx] += dth
        th_new = np.minimum(np.maximum(th_new, lower_lim), upper_lim)
        bl_new = np.clip(bl_vec_base + dbl, bone_clip[0], bone_clip[1]) if bl_vec_base.size else bl_vec_base
        r_new = root_base.copy()
        if optimize_root:
            r_new = root_base + droot
        return th_new, bl_new, r_new

    def _eval_err_batch(th_cand, bl_cand, root_cand, corr_eval, idx_batch):
        jp_cand, _ = fk_positions(th_cand, bl_cand, root_cand)
        if corr_eval.get('mode','hard') == 'hard':
            mb = [corr_eval['hard'][i] for i in idx_batch]
            if fast_vectorized:
                e_cand = build_residual_stack_hard_geom_vec(jp_cand, markers[idx_batch], mb, geom=geom, bone_radii=bone_radii)
            else:
                e_cand = build_residual_stack_hard_geom(jp_cand, markers[idx_batch], mb, geom=geom, bone_radii=bone_radii)
        else:
            cands_sub   = [corr_eval['cands'][i]   for i in idx_batch]
            weights_sub = [corr_eval['weights'][i] for i in idx_batch]
            e_cand = build_residual_stack_soft_geom(jp_cand, markers[idx_batch], BONES_IDX, cands_sub, weights_sub, geom=geom, bone_radii=bone_radii)
        w_sqrt_batch = w_sqrt_full[idx_batch]
        ew = np.repeat(w_sqrt_batch, 3) * e_cand
        err = np.linalg.norm(ew)
        cost = 0.5 * float(ew.T @ ew)
        return err, cost

    assign_state = {'hard': None, 'weights': None}
    jp, bl_all = fk_positions(theta, bl_vec, root)

    # initial correspondences
    if auto_assign_bones or (marker_bones is None):
        corr = robust_assign_markers(
            markers, jp, BONES_IDX, use_segment=True, prev_state=assign_state,
            bone_lengths=bl_all,
            topk=assign_topk, soft_sigma_factor=assign_soft_sigma_factor,
            distance_gate_abs=assign_distance_gate_abs, distance_gate_factor=assign_distance_gate_factor,
            enable_gate=assign_enable_gate,
            hysteresis_margin=assign_hysteresis_margin, enable_hysteresis=assign_enable_hysteresis,
            temporal_smoothing=assign_temporal_smoothing, enable_temporal_smoothing=assign_enable_temporal_smoothing,
            semantic_priors=assign_semantic_priors,
            geom=geom, bone_radii=bone_radii
        )
        assign_state = corr['state']
    else:
        corr = {'mode': 'hard', 'hard': list(marker_bones)}

    idx_batch = _pick_indices()
    prev_err, _ = _eval_err_batch(theta, bl_vec, root, corr, idx_batch)

    for it in range(max_iters):
        jp_base, bl_all = fk_positions(theta, bl_vec, root)
        if (auto_assign_bones or (marker_bones is None)) and ((it % reassign_every) == 0):
            corr = robust_assign_markers(
                markers, jp_base, BONES_IDX, use_segment=True, prev_state=assign_state,
                bone_lengths=bl_all, topk=assign_topk, soft_sigma_factor=assign_soft_sigma_factor,
                distance_gate_abs=assign_distance_gate_abs, distance_gate_factor=assign_distance_gate_factor,
                enable_gate=assign_enable_gate, hysteresis_margin=assign_hysteresis_margin,
                enable_hysteresis=assign_enable_hysteresis, temporal_smoothing=assign_temporal_smoothing,
                enable_temporal_smoothing=assign_enable_temporal_smoothing, semantic_priors=assign_semantic_priors,
                geom=geom, bone_radii=bone_radii
            )
            assign_state = corr['state']

        idx_batch = _pick_indices()
        Kb = len(idx_batch)

        # Build residual on batch
        if corr.get('mode', 'hard') == 'hard':
            if fast_vectorized:
                e = build_residual_stack_hard_geom_vec(jp_base, markers[idx_batch], [corr['hard'][i] for i in idx_batch], geom=geom, bone_radii=bone_radii)
            else:
                e = build_residual_stack_hard_geom(jp_base, markers[idx_batch], [corr['hard'][i] for i in idx_batch], geom=geom, bone_radii=bone_radii)
        else:
            cands_sub   = [corr['cands'][i]   for i in idx_batch]
            weights_sub = [corr['weights'][i] for i in idx_batch]
            e = build_residual_stack_soft_geom(jp_base, markers[idx_batch], BONES_IDX, cands_sub, weights_sub, geom=geom, bone_radii=bone_radii)

        # Jacobians
        n_bones = bl_vec.size
        J_theta = np.zeros((3 * Kb, n_active))
        J_bl    = np.zeros((3 * Kb, n_bones))
        J_root  = np.zeros((3 * Kb, 3 if optimize_root else 0))

        for c, j_idx in enumerate(active_angle_idx):
            orig = theta[j_idx]
            theta[j_idx] = orig + angle_delta
            jp_pert, _ = fk_positions(theta, bl_vec, root)
            if corr.get('mode', 'hard') == 'hard':
                e_pert = (build_residual_stack_hard_geom_vec(jp_pert, markers[idx_batch],
                          [corr['hard'][i] for i in idx_batch], geom=geom, bone_radii=bone_radii)
                          if fast_vectorized else
                          build_residual_stack_hard_geom(jp_pert, markers[idx_batch],
                          [corr['hard'][i] for i in idx_batch], geom=geom, bone_radii=bone_radii))
            else:
                cands_sub   = [corr['cands'][i]   for i in idx_batch]
                weights_sub = [corr['weights'][i] for i in idx_batch]
                e_pert = build_residual_stack_soft_geom(jp_pert, markers[idx_batch], BONES_IDX, cands_sub, weights_sub, geom=geom, bone_radii=bone_radii)
            J_theta[:, c] = (e_pert - e) / angle_delta
            theta[j_idx] = orig

        for c in range(n_bones):
            orig = bl_vec[c]
            bl_vec[c] = orig + length_delta
            jp_pert, _ = fk_positions(theta, bl_vec, root)
            if corr.get('mode', 'hard') == 'hard':
                e_pert = (build_residual_stack_hard_geom_vec(jp_pert, markers[idx_batch],
                          [corr['hard'][i] for i in idx_batch], geom=geom, bone_radii=bone_radii)
                          if fast_vectorized else
                          build_residual_stack_hard_geom(jp_pert, markers[idx_batch],
                          [corr['hard'][i] for i in idx_batch], geom=geom, bone_radii=bone_radii))
            else:
                cands_sub   = [corr['cands'][i]   for i in idx_batch]
                weights_sub = [corr['weights'][i] for i in idx_batch]
                e_pert = build_residual_stack_soft_geom(jp_pert, markers[idx_batch], BONES_IDX, cands_sub, weights_sub, geom=geom, bone_radii=bone_radii)
            J_bl[:, c] = (e_pert - e) / length_delta
            bl_vec[c] = orig

        if optimize_root:
            for c in range(3):
                root[c] += root_delta
                jp_pert, _ = fk_positions(theta, bl_vec, root)
                if corr.get('mode', 'hard') == 'hard':
                    e_pert = (build_residual_stack_hard_geom_vec(jp_pert, markers[idx_batch],
                              [corr['hard'][i] for i in idx_batch], geom=geom, bone_radii=bone_radii)
                              if fast_vectorized else
                              build_residual_stack_hard_geom(jp_pert, markers[idx_batch],
                              [corr['hard'][i] for i in idx_batch], geom=geom, bone_radii=bone_radii))
                else:
                    cands_sub   = [corr['cands'][i]   for i in idx_batch]
                    weights_sub = [corr['weights'][i] for i in idx_batch]
                    e_pert = build_residual_stack_soft_geom(jp_pert, markers[idx_batch], BONES_IDX, cands_sub, weights_sub, geom=geom, bone_radii=bone_radii)
                J_root[:, c] = (e_pert - e) / root_delta
                root[c] -= root_delta

        # Stack Jacobian + weights
        blocks = [J_theta, J_bl]
        if optimize_root: blocks.append(J_root)
        J = np.hstack(blocks)

        e_weighted = e.copy()
        w_sqrt_batch = w_sqrt_full[idx_batch]
        if not np.allclose(w_sqrt_batch, 1.0):
            for k in range(Kb):
                row = 3 * k
                J[row:row+3, :] *= w_sqrt_batch[k]
                e_weighted[row:row+3] *= w_sqrt_batch[k]

        JTJ = J.T @ J
        JTe = - (J.T @ e_weighted)
        reg_vec = np.concatenate([
            angle_reg * np.ones(n_active),
            bone_reg  * np.ones(bl_vec.size),
            (root_reg * np.ones(3) if optimize_root else np.zeros(0)),
        ])
        D = np.diag(reg_vec)
        prev_cost = 0.5 * float((e_weighted.T @ e_weighted))
        improved = False

        if strategy == "lm":
            for _trial in range(8):
                A = JTJ + (lm_lambda ** 2) * D
                try:
                    L = np.linalg.cholesky(A)
                    y = np.linalg.solve(L, JTe)
                    delta = np.linalg.solve(L.T, y)
                except np.linalg.LinAlgError:
                    delta = np.linalg.solve(A, JTe)
                d_theta, d_bl, d_root = _split(delta)
                d_theta, d_bl, d_root = _clip_steps(d_theta, d_bl, d_root)
                theta_new, bl_vec_new, root_new = _propose(theta, bl_vec, root, d_theta, d_bl, d_root)

                corr_eval = corr
                if allow_trial_reassign and (auto_assign_bones or (marker_bones is None)):
                    jp_tmp, bl_tmp = fk_positions(theta_new, bl_vec_new, root_new)
                    corr_eval = robust_assign_markers(
                        markers, jp_tmp, BONES_IDX, use_segment=True, prev_state=assign_state,
                        bone_lengths=bl_tmp, topk=assign_topk, soft_sigma_factor=assign_soft_sigma_factor,
                        distance_gate_abs=assign_distance_gate_abs, distance_gate_factor=assign_distance_gate_factor,
                        enable_gate=assign_enable_gate, hysteresis_margin=assign_hysteresis_margin,
                        enable_hysteresis=assign_enable_hysteresis, temporal_smoothing=assign_temporal_smoothing,
                        enable_temporal_smoothing=assign_enable_temporal_smoothing, semantic_priors=assign_semantic_priors,
                        geom=geom, bone_radii=bone_radii
                    )

                err_new, _ = _eval_err_batch(theta_new, bl_vec_new, root_new, corr_eval, idx_batch)
                if verbose:
                    print(f"iter {it:03d} LM trial: err_new={err_new:.6f}, prev_err={prev_err:.6f}, λ={lm_lambda:.2e}")
                if err_new < prev_err:
                    theta, bl_vec, root = theta_new, bl_vec_new, root_new
                    prev_err = err_new
                    lm_lambda = max(lm_lambda / lm_lambda_factor, lm_lambda_min)
                    improved = True
                    break
                else:
                    lm_lambda = min(lm_lambda * lm_lambda_factor, lm_lambda_max)

        elif strategy == "lm+linesearch":
            A = JTJ + (lm_lambda ** 2) * D
            try:
                L = np.linalg.cholesky(A)
                y = np.linalg.solve(L, JTe)
                delta = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                delta = np.linalg.solve(A, JTe)

            d_theta_base, d_bl_base, d_root_base = _split(delta)
            for s in line_search_scales:
                d_theta, d_bl, d_root = _clip_steps(d_theta_base * s, d_bl_base * s, d_root_base * s)
                theta_new, bl_vec_new, root_new = _propose(theta, bl_vec, root, d_theta, d_bl, d_root)
                corr_eval = corr
                if allow_trial_reassign and (auto_assign_bones or (marker_bones is None)):
                    jp_tmp, bl_tmp = fk_positions(theta_new, bl_vec_new, root_new)
                    corr_eval = robust_assign_markers(
                        markers, jp_tmp, BONES_IDX, use_segment=True, prev_state=assign_state,
                        bone_lengths=bl_tmp, topk=assign_topk, soft_sigma_factor=assign_soft_sigma_factor,
                        distance_gate_abs=assign_distance_gate_abs, distance_gate_factor=assign_distance_gate_factor,
                        enable_gate=assign_enable_gate, hysteresis_margin=assign_hysteresis_margin,
                        enable_hysteresis=assign_enable_hysteresis, temporal_smoothing=assign_temporal_smoothing,
                        enable_temporal_smoothing=assign_enable_temporal_smoothing, semantic_priors=assign_semantic_priors,
                        geom=geom, bone_radii=bone_radii
                    )
                err_new, _ = _eval_err_batch(theta_new, bl_vec_new, root_new, corr_eval, idx_batch)
                if verbose:
                    print(f"iter {it:03d} LS s={s:.3f}: err_new={err_new:.6f}, prev_err={prev_err:.6f}")
                if err_new < prev_err:
                    theta, bl_vec, root = theta_new, bl_vec_new, root_new
                    prev_err = err_new
                    improved = True
                    break

        elif strategy == "lm+dogleg":
            g = J.T @ e_weighted
            B = JTJ
            try:
                p_gn = np.linalg.solve(B + 1e-9 * np.eye(B.shape[0]), -g)
            except np.linalg.LinAlgError:
                p_gn = -g
            denom = float(g.T @ (B @ g)) + 1e-12
            alpha = float((g.T @ g) / denom)
            p_sd = -alpha * g

            def dogleg(p_sd, p_gn, Delta):
                n_sd = np.linalg.norm(p_sd); n_gn = np.linalg.norm(p_gn)
                if n_gn <= Delta: return p_gn
                if n_sd >= Delta: return p_sd * (Delta / (n_sd + 1e-12))
                d = p_gn - p_sd
                a = float(d.T @ d)
                b = 2.0 * float(p_sd.T @ d)
                c = float(p_sd.T @ p_sd) - Delta**2
                disc = max(0.0, b*b - 4*a*c)
                t = (-b + np.sqrt(disc)) / (2*a + 1e-12); t = np.clip(t, 0.0, 1.0)
                return p_sd + t * d

            tries = 0
            while tries < 6:
                p = dogleg(p_sd, p_gn, tr_radius)
                d_theta, d_bl, d_root = _split(p)
                d_theta, d_bl, d_root = _clip_steps(d_theta, d_bl, d_root)
                theta_new, bl_vec_new, root_new = _propose(theta, bl_vec, root, d_theta, d_bl, d_root)
                err_new, new_cost = _eval_err_batch(theta_new, bl_vec_new, root_new, corr, idx_batch)
                pred_red = - (float(g.T @ p) + 0.5 * float(p.T @ (B @ p)))
                rho = (prev_cost - new_cost) / (pred_red + 1e-12)
                if verbose:
                    print(f"iter {it:03d} dogleg: radius={tr_radius:.3f}, rho={rho:.3f}, err_new={err_new:.6f}, prev_err={prev_err:.6f}")
                if (rho >= tr_eta) and (new_cost < prev_cost):
                    theta, bl_vec, root = theta_new, bl_vec_new, root_new
                    prev_err = err_new
                    improved = True
                    if rho > 0.75 and np.linalg.norm(p) > 0.9 * tr_radius:
                        tr_radius = min(tr_radius * tr_expand, tr_radius_max)
                    elif rho < 0.25:
                        tr_radius = max(tr_radius * tr_shrink, 1e-4)
                    break
                else:
                    tr_radius = max(tr_radius * tr_shrink, 1e-4)
                    tries += 1
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        angles_history.append(theta.copy())
        bone_length_history.append(bl_vec.copy())
        root_history.append(root.copy())

        if verbose:
            print(f"[LM markers] iter {it+1:03d}  batch_err={prev_err:.6f}")
        if prev_err < tolerance:
            if verbose:
                print(f"[LM markers] converged in {it+1} iters, batch_err={prev_err:.6f}")
            break

    bone_lengths_final = bone_lengths.copy()
    for k, v in zip(bl_keys, bl_vec):
        bone_lengths_final[k] = v
    return theta, bone_lengths_final, root, angles_history, bone_length_history, root_history

# ====================== #
# Hinge presets & GT     #
# ====================== #
def make_active_dof_indices_human_like_hinges():
    """Allow torso/neck, shoulders, hips (full 3-DOF) + elbows/knees pitch-only."""
    active = []
    active += [ SPINE_TOP_YAW, SPINE_TOP_PITCH, SPINE_TOP_ROLL,
                 NECK_TOP_PITCH,  NECK_TOP_ROLL]
    active += [RIGHT_SHOULDER_YAW, RIGHT_SHOULDER_ROLL, RIGHT_ELBOW_YAW,
               LEFT_SHOULDER_YAW,  LEFT_SHOULDER_ROLL, LEFT_ELBOW_YAW]
    active += [RIGHT_HIP_YAW, RIGHT_HIP_PITCH, RIGHT_HIP_ROLL,
               LEFT_HIP_YAW,  LEFT_HIP_PITCH,  LEFT_HIP_ROLL]
    active += [RIGHT_ELBOW_PITCH, LEFT_ELBOW_PITCH, RIGHT_KNEE_PITCH, LEFT_KNEE_PITCH]
    return sorted(set(active))

def enforce_pure_hinges_in_limits(lower, upper, tight_deg=0.5):
    eps = np.deg2rad(tight_deg)
    for j in (RIGHT_ELBOW, LEFT_ELBOW, RIGHT_KNEE, LEFT_KNEE):
        i = 3 * j
        lower[i + 0], upper[i + 0] = -eps, eps
        lower[i + 2], upper[i + 2] = -eps, eps
    return lower, upper

# --------- Simple gait generator (angles + root translation) -----------------
def gait_angles(t, f=1.2):
    """
    Returns a 48-dim angle vector for time t (seconds).
    Simple sinusoidal gait: hips & shoulders out of phase; knees flex on stance.
    """
    w = 2.0 * np.pi * f
    th = get_default_joint_angles()

    # Torso/neck micro motion
    th[SPINE_TOP_YAW]   = np.deg2rad(5.0) * np.sin(w * t * 0.5)
    th[SPINE_TOP_PITCH] = np.deg2rad(3.0) * np.sin(w * t + np.pi/2)
    th[SPINE_TOP_ROLL]  = np.deg2rad(2.0) * np.sin(w * t)

    th[NECK_TOP_PITCH]  = np.deg2rad(2.0) * np.sin(w * t + np.pi/2)
    th[NECK_TOP_ROLL]   = np.deg2rad(2.0) * np.sin(w * t)

    # Legs
    A_hip   = np.deg2rad(30.0)
    A_knee  = np.deg2rad(45.0)
    hipR =  A_hip * np.sin(w * t)
    hipL = -A_hip * np.sin(w * t)
    kneeR = 0.5 * A_knee * (1.0 - np.cos(w * t))       # >= 0
    kneeL = 0.5 * A_knee * (1.0 - np.cos(w * t + np.pi))
    th[RIGHT_HIP_PITCH]  = hipR
    th[LEFT_HIP_PITCH]   = hipL
    th[RIGHT_KNEE_PITCH] = kneeR
    th[LEFT_KNEE_PITCH]  = kneeL

    # Arms (counter-swing)
    A_sh_yaw  = np.deg2rad(25.0)
    A_sh_roll = np.deg2rad(15.0)
    A_elbow   = np.deg2rad(25.0)
    shYawR =  A_sh_yaw * np.sin(w * t + np.pi)
    shYawL =  A_sh_yaw * np.sin(w * t)
    shRollR = -A_sh_roll * np.sin(w * t + np.pi)
    shRollL =  A_sh_roll * np.sin(w * t)
    elbR = 0.5 * A_elbow * (1.0 - np.cos(w * t + np.pi))
    elbL = 0.5 * A_elbow * (1.0 - np.cos(w * t))
    th[RIGHT_SHOULDER_YAW] = shYawR
    th[RIGHT_SHOULDER_ROLL] = shRollR
    th[LEFT_SHOULDER_YAW] = shYawL
    th[LEFT_SHOULDER_ROLL] = shRollL
    th[RIGHT_ELBOW_PITCH] = elbR
    th[LEFT_ELBOW_PITCH]  = elbL
    return th

def gait_root(t, speed=1.0, f=1.2):
    """
    Pelvis world translation over time.
    - forward X at 'speed' m/s
    - small lateral Y sway and vertical Z bounce
    """
    w = 2.0 * np.pi * f
    x = speed * t
    y = 0.03 * np.sin(w * t + np.pi/2)
    z = 0.02 * np.sin(2.0 * w * t)  # small bounce at 2*f
    return np.array([x, y, z], dtype=float)

def save_animation(ani, filename_base, fps=30, dpi=150):
    # Try FFmpeg first
    try:
        # Optional: use bundled ffmpeg if available
        try:
            import imageio_ffmpeg
            mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

        writer = FFMpegWriter(fps=fps, metadata={'artist': 'Me'}, bitrate=1800)
        ani.save(f"{filename_base}.mp4", writer=writer, dpi=dpi)
        print(f"Saved {filename_base}.mp4 with FFmpeg")
        return
    except Exception as e:
        print(f"FFmpeg not available or failed ({e}). Falling back to GIF...")

    # Fallback: GIF with Pillow
    try:
        writer = PillowWriter(fps=fps)
        ani.save(f"{filename_base}.gif", writer=writer, dpi=dpi)
        print(f"Saved {filename_base}.gif with PillowWriter")
    except Exception as e:
        print(f"PillowWriter also failed: {e}")
        print("As a last resort, save frames and stitch externally.")

# ---------- Motion library: more gaits / motions ----------
def _deg(a): return np.deg2rad(a)

def angles_walk(t, f=1.2,
                A_hip_deg=30, A_knee_deg=45,
                A_sh_yaw_deg=25, A_sh_roll_deg=15, A_elbow_deg=25,
                foot_push_deg=10):
    w = 2.0*np.pi*f
    th = get_default_joint_angles()

    # torso/neck micro
    th[SPINE_TOP_YAW]   = _deg(5.0) * np.sin(0.5*w*t)
    th[SPINE_TOP_PITCH] = _deg(3.0) * np.sin(w*t + np.pi/2)
    th[SPINE_TOP_ROLL]  = _deg(2.0) * np.sin(w*t)
    th[NECK_TOP_PITCH]  = _deg(2.0) * np.sin(w*t + np.pi/2)
    th[NECK_TOP_ROLL]   = _deg(2.0) * np.sin(w*t)

    # legs (π phase offset)
    A_hip  = _deg(A_hip_deg)
    A_knee = _deg(A_knee_deg)
    hipR   =  A_hip * np.sin(w*t)
    hipL   = -A_hip * np.sin(w*t)
    kneeR  = 0.5 * A_knee * (1.0 - np.cos(w*t))
    kneeL  = 0.5 * A_knee * (1.0 - np.cos(w*t + np.pi))
    th[RIGHT_HIP_PITCH]   = hipR
    th[LEFT_HIP_PITCH]    = hipL
    th[RIGHT_KNEE_PITCH]  = kneeR
    th[LEFT_KNEE_PITCH]   = kneeL

    # simple toe-off / heel-strike with feet
    push = _deg(foot_push_deg)
    # th[RIGHT_FOOT_PITCH] =  0.5*push*np.sin(w*t + 0.2*np.pi)
    # th[LEFT_FOOT_PITCH]  =  0.5*push*np.sin(w*t + 0.2*np.pi + np.pi)

    # arms (counter-swing)
    A_sh_yaw  = _deg(A_sh_yaw_deg)
    A_sh_roll = _deg(A_sh_roll_deg)
    A_elbow   = _deg(A_elbow_deg)
    th[RIGHT_SHOULDER_YAW]  =  A_sh_yaw  * np.sin(w*t + np.pi)
    th[LEFT_SHOULDER_YAW]   =  A_sh_yaw  * np.sin(w*t)
    th[RIGHT_SHOULDER_ROLL] = -A_sh_roll * np.sin(w*t + np.pi)
    th[LEFT_SHOULDER_ROLL]  =  A_sh_roll * np.sin(w*t)
    th[RIGHT_ELBOW_PITCH]   = 0.5 * A_elbow * (1.0 - np.cos(w*t + np.pi))
    th[LEFT_ELBOW_PITCH]    = 0.5 * A_elbow * (1.0 - np.cos(w*t))
    return th

def root_walk(t, speed=1.0, f=1.2):
    w = 2.0*np.pi*f
    x = speed * t
    y = 0.03 * np.sin(w*t + np.pi/2)
    z = 0.02 * np.sin(2*w*t)
    return np.array([x,y,z], float)

def angles_run(t, f=2.4,
               A_hip_deg=45, A_knee_deg=75,
               A_sh_yaw_deg=40, A_elbow_deg=40,
               pelvis_pitch_deg=5, foot_push_deg=18):
    w = 2*np.pi*f
    th = get_default_joint_angles()
    # lean
    th[PELVIS + 1] = _deg(pelvis_pitch_deg)  # pelvis pitch forward

    # legs bigger amplitudes
    A_hip, A_knee = _deg(A_hip_deg), _deg(A_knee_deg)
    hipR =  A_hip * np.sin(w*t)
    hipL = -A_hip * np.sin(w*t)
    kneeR = 0.5*A_knee*(1.0 - np.cos(w*t))
    kneeL = 0.5*A_knee*(1.0 - np.cos(w*t + np.pi))
    th[RIGHT_HIP_PITCH], th[LEFT_HIP_PITCH] = hipR, hipL
    th[RIGHT_KNEE_PITCH], th[LEFT_KNEE_PITCH] = kneeR, kneeL

    # feet: stronger toe-off
    # push = _deg(foot_push_deg)
    # th[RIGHT_FOOT_PITCH] = push*np.maximum(0.0, np.sin(w*t - 0.2*np.pi))
    # th[LEFT_FOOT_PITCH]  = push*np.maximum(0.0, np.sin(w*t - 0.2*np.pi + np.pi))

    # arms
    A_sh_yaw = _deg(A_sh_yaw_deg)
    A_elb    = _deg(A_elbow_deg)
    th[RIGHT_SHOULDER_YAW] =  A_sh_yaw*np.sin(w*t + np.pi)
    th[LEFT_SHOULDER_YAW]  =  A_sh_yaw*np.sin(w*t)
    th[RIGHT_ELBOW_PITCH]  = 0.6*A_elb*(1.0 - np.cos(w*t + np.pi))
    th[LEFT_ELBOW_PITCH]   = 0.6*A_elb*(1.0 - np.cos(w*t))
    return th

def root_run(t, speed=3.0, f=2.4):
    w = 2*np.pi*f
    x = speed * t
    y = 0.04 * np.sin(w*t + np.pi/2)
    z = 0.05 * np.maximum(0.0, np.sin(2*w*t))  # crude "flight" bounce
    return np.array([x,y,z], float)

def angles_march(t, f=1.5, lift_deg=80, arm_deg=35, foot_dorsi_deg=15):
    w = 2*np.pi*f
    th = get_default_joint_angles()
    A_hip = _deg(lift_deg)
    kneeR = 0.3*_deg(lift_deg)*(1.0 - np.cos(w*t))
    kneeL = 0.3*_deg(lift_deg)*(1.0 - np.cos(w*t + np.pi))
    th[RIGHT_HIP_PITCH] =  A_hip*np.maximum(0.0, np.sin(w*t))
    th[LEFT_HIP_PITCH]  =  A_hip*np.maximum(0.0, np.sin(w*t + np.pi))
    th[RIGHT_KNEE_PITCH], th[LEFT_KNEE_PITCH] = kneeR, kneeL
    # dorsiflex during swing
    # th[RIGHT_FOOT_PITCH] = -_deg(foot_dorsi_deg)*np.maximum(0.0, np.sin(w*t))
    # th[LEFT_FOOT_PITCH]  = -_deg(foot_dorsi_deg)*np.maximum(0.0, np.sin(w*t + np.pi))
    # arms big
    A_arm = _deg(arm_deg)
    th[RIGHT_SHOULDER_YAW] =  A_arm*np.sin(w*t + np.pi)
    th[LEFT_SHOULDER_YAW]  =  A_arm*np.sin(w*t)
    return th

def root_march(t, speed=0.8, f=1.5):
    w = 2*np.pi*f
    return np.array([speed*t, 0.02*np.sin(w*t+np.pi/2), 0.02*np.sin(2*w*t)], float)

def angles_sidestep(t, f=1.2, hip_roll_deg=22, knee_deg=10, arm_counter_deg=12):
    w = 2*np.pi*f
    th = get_default_joint_angles()
    # abduct/adduct via hip roll; knees soft
    Aroll = _deg(hip_roll_deg)
    th[RIGHT_HIP_ROLL] = -Aroll*np.sin(w*t)
    th[LEFT_HIP_ROLL]  =  Aroll*np.sin(w*t)
    th[RIGHT_KNEE_PITCH] = _deg(knee_deg)*(0.5 - 0.5*np.cos(w*t))
    th[LEFT_KNEE_PITCH]  = _deg(knee_deg)*(0.5 - 0.5*np.cos(w*t + np.pi))
    # arms counter-roll
    th[RIGHT_SHOULDER_ROLL] =  _deg(arm_counter_deg)*np.sin(w*t)
    th[LEFT_SHOULDER_ROLL]  = -_deg(arm_counter_deg)*np.sin(w*t)
    return th

def root_sidestep(t, speed_lat=0.6, f=1.2):
    w = 2*np.pi*f
    x = 0.0
    y = speed_lat * t
    z = 0.02*np.sin(2*w*t)
    return np.array([x,y,z], float)

def angles_turn_in_place(t, turn_rate_deg_s=60, knee_soft_deg=8):
    th = get_default_joint_angles()
    # pelvis yaw accumulates over time
    th[PELVIS + 0] = _deg(turn_rate_deg_s) * t
    # soft knee flexion to look natural
    th[RIGHT_KNEE_PITCH] = _deg(knee_soft_deg)*(0.5 - 0.5*np.cos(2.0*np.pi*0.8*t))
    th[LEFT_KNEE_PITCH]  = _deg(knee_soft_deg)*(0.5 - 0.5*np.cos(2.0*np.pi*0.8*t + np.pi))
    return th

def root_turn_in_place(t):
    return np.zeros(3, float)

def angles_squat(t, f=0.6, depth_deg=85, hip_back_deg=35, arms_fwd_deg=20):
    w = 2*np.pi*f
    th = get_default_joint_angles()
    squat = 0.5*(1.0 - np.cos(w*t))   # 0..1..0
    th[RIGHT_KNEE_PITCH] = squat*_deg(depth_deg)
    th[LEFT_KNEE_PITCH]  = squat*_deg(depth_deg)
    th[RIGHT_HIP_PITCH]  = -squat*_deg(hip_back_deg)
    th[LEFT_HIP_PITCH]   = -squat*_deg(hip_back_deg)
    th[RIGHT_SHOULDER_PITCH] = -squat*_deg(arms_fwd_deg) if 'RIGHT_SHOULDER_PITCH' in globals() else 0.0
    th[LEFT_SHOULDER_PITCH]  = -squat*_deg(arms_fwd_deg) if 'LEFT_SHOULDER_PITCH'  in globals() else 0.0
    return th

def root_squat(t, f=0.6, drop=0.12):
    w = 2*np.pi*f
    z = -drop * 0.5*(1.0 - np.cos(w*t))
    return np.array([0.0, 0.0, z], float)

def angles_jump(t, f=1.0, crouch_deg=70, arm_swing_deg=60):
    w = 2*np.pi*f
    th = get_default_joint_angles()
    # crouch → extend → land (sinusoidal)
    phase = 0.5*(1.0 - np.cos(w*t))  # 0..1..0
    th[RIGHT_KNEE_PITCH] = phase*_deg(crouch_deg)
    th[LEFT_KNEE_PITCH]  = phase*_deg(crouch_deg)
    th[RIGHT_HIP_PITCH]  = -0.5*phase*_deg(crouch_deg)
    th[LEFT_HIP_PITCH]   = -0.5*phase*_deg(crouch_deg)
    th[RIGHT_SHOULDER_YAW] =  _deg(arm_swing_deg)*(np.sin(w*t + np.pi))
    th[LEFT_SHOULDER_YAW]  =  _deg(arm_swing_deg)*np.sin(w*t)
    return th

def root_jump(t, f=1.0, rise=0.12):
    # crude parabola-like vertical hop (no proper flight dynamics)
    w = 2*np.pi*f
    z = rise * np.maximum(0.0, np.sin(w*t))**2
    return np.array([0.0, 0.0, z], float)

def angles_stairs(t, f=1.2, knee_deg=65, hip_deg=25, foot_dorsi_deg=18):
    w = 2*np.pi*f
    th = get_default_joint_angles()
    # similar to walk but with higher knee and dorsiflexion on swing
    A_hip, A_knee = _deg(hip_deg), _deg(knee_deg)
    th[RIGHT_HIP_PITCH]  =  A_hip*np.sin(w*t)
    th[LEFT_HIP_PITCH]   = -A_hip*np.sin(w*t)
    th[RIGHT_KNEE_PITCH] = 0.5*A_knee*(1.0 - np.cos(w*t))
    th[LEFT_KNEE_PITCH]  = 0.5*A_knee*(1.0 - np.cos(w*t + np.pi))
    # th[RIGHT_FOOT_PITCH] = -_deg(foot_dorsi_deg)*np.maximum(0.0, np.sin(w*t))
    # th[LEFT_FOOT_PITCH]  = -_deg(foot_dorsi_deg)*np.maximum(0.0, np.sin(w*t + np.pi))
    return th

def root_stairs(t, speed=0.9, f=1.2, step_h=0.16):
    # forward plus very slight upward drift per cycle to suggest ascent
    cycles = f * t
    x = speed * t
    y = 0.0
    z = 0.02*np.sin(2*np.pi*2*f*t) + step_h * 0.1 * cycles
    return np.array([x,y,z], float)

def make_motion(kind, **kw):
    kind = kind.lower()
    if kind == "walk":
        return (lambda t: angles_walk(t, f=kw.get("f",1.2),
                                      A_hip_deg=kw.get("A_hip_deg",30),
                                      A_knee_deg=kw.get("A_knee_deg",45),
                                      A_sh_yaw_deg=kw.get("A_sh_yaw_deg",25),
                                      A_sh_roll_deg=kw.get("A_sh_roll_deg",15),
                                      A_elbow_deg=kw.get("A_elbow_deg",25),
                                      foot_push_deg=kw.get("foot_push_deg",10)),
                lambda t: root_walk(t, speed=kw.get("speed",1.0), f=kw.get("f",1.2)))
    if kind == "run":
        return (lambda t: angles_run(t, f=kw.get("f",2.4),
                                     A_hip_deg=kw.get("A_hip_deg",45),
                                     A_knee_deg=kw.get("A_knee_deg",75),
                                     A_sh_yaw_deg=kw.get("A_sh_yaw_deg",40),
                                     A_elbow_deg=kw.get("A_elbow_deg",40),
                                     pelvis_pitch_deg=kw.get("pelvis_pitch_deg",5),
                                     foot_push_deg=kw.get("foot_push_deg",18)),
                lambda t: root_run(t, speed=kw.get("speed",3.0), f=kw.get("f",2.4)))
    if kind == "march":
        return (lambda t: angles_march(t, f=kw.get("f",1.5),
                                       lift_deg=kw.get("lift_deg",80),
                                       arm_deg=kw.get("arm_deg",35),
                                       foot_dorsi_deg=kw.get("foot_dorsi_deg",15)),
                lambda t: root_march(t, speed=kw.get("speed",0.8), f=kw.get("f",1.5)))
    if kind == "sidestep":
        return (lambda t: angles_sidestep(t, f=kw.get("f",1.2),
                                          hip_roll_deg=kw.get("hip_roll_deg",22),
                                          knee_deg=kw.get("knee_deg",10),
                                          arm_counter_deg=kw.get("arm_counter_deg",12)),
                lambda t: root_sidestep(t, speed_lat=kw.get("speed_lat",0.6), f=kw.get("f",1.2)))
    if kind == "turn":
        return (lambda t: angles_turn_in_place(t, turn_rate_deg_s=kw.get("turn_rate_deg_s",60),
                                               knee_soft_deg=kw.get("knee_soft_deg",8)),
                lambda t: root_turn_in_place(t))
    if kind == "squat":
        return (lambda t: angles_squat(t, f=kw.get("f",0.6),
                                       depth_deg=kw.get("depth_deg",85),
                                       hip_back_deg=kw.get("hip_back_deg",35),
                                       arms_fwd_deg=kw.get("arms_fwd_deg",20)),
                lambda t: root_squat(t, f=kw.get("f",0.6), drop=kw.get("drop",0.12)))
    if kind == "jump":
        return (lambda t: angles_jump(t, f=kw.get("f",1.0),
                                      crouch_deg=kw.get("crouch_deg",70),
                                      arm_swing_deg=kw.get("arm_swing_deg",60)),
                lambda t: root_jump(t, f=kw.get("f",1.0), rise=kw.get("rise",0.12)))
    if kind == "stairs":
        return (lambda t: angles_stairs(t, f=kw.get("f",1.2),
                                        knee_deg=kw.get("knee_deg",65),
                                        hip_deg=kw.get("hip_deg",25),
                                        foot_dorsi_deg=kw.get("foot_dorsi_deg",18)),
                lambda t: root_stairs(t, speed=kw.get("speed",0.9), f=kw.get("f",1.2),
                                      step_h=kw.get("step_h",0.16)))
    raise ValueError(f"Unknown motion kind: {kind}. Try one of: walk, run, march, sidestep, turn, squat, jump, stairs.")


def _read_table(path: str):
    if pd is None:
        # minimal reader: expects comma-separated with a header
        import csv
        with open(path, "r", newline="") as f:
            rows = list(csv.reader(f))
        header = [h.strip() for h in rows[0]]
        data = [[float(x) if x.replace('.','',1).replace('-','',1).isdigit() else x for x in r] for r in rows[1:]]
        # build a dict-like "lite df"
        class _Lite:
            def __init__(self, header, data):
                self.columns = header
                self._data = data
            def to_numpy(self): return np.array(self._data, dtype=object)
            def __getitem__(self, key):
                i = self.columns.index(key)
                return np.array([row[i] for row in self._data], dtype=object)
        return _Lite(header, data)
    else:
        return pd.read_csv(path, sep=None, engine="python")

def _detect_format(df) -> str:
    cols = set(c.lower() for c in (df.columns if pd is not None else df.columns))
    if {"frame", "name", "x", "y", "z"}.issubset(cols):
        return "long"
    pat = re.compile(r"(.+)_([xyz])$", flags=re.IGNORECASE)
    has_xyz = any(pat.match(c) for c in (df.columns if pd is not None else df.columns)
                  if c.lower() not in ("frame", "time", "time_s"))
    return "wide" if has_xyz else "unknown"

def _extract_from_long(df):
    # normalize columns
    if pd is not None:
        d = df.rename(columns={c: c.lower() for c in df.columns})
        if "time" in d.columns and "time_s" not in d.columns:
            d = d.rename(columns={"time": "time_s"})
        names = sorted(d["name"].unique().tolist())
        frames = np.sort(d["frame"].unique())
        F, M = len(frames), len(names)
        name_to_idx = {n: i for i, n in enumerate(names)}
        # times
        times = (d.groupby("frame")["time_s"].first().reindex(frames).to_numpy()
                 if "time_s" in d.columns else np.arange(F, dtype=float))
        XYZ = np.zeros((F, M, 3), dtype=float)
        g = d.groupby("frame")
        for fi, fr in enumerate(frames):
            sub = g.get_group(fr).drop_duplicates(subset=["name"], keep="last").set_index("name")
            for n, row in sub.iterrows():
                if n in name_to_idx:
                    mi = name_to_idx[n]
                    XYZ[fi, mi, :] = [row["x"], row["y"], row["z"]]
        return XYZ, names, times
    else:
        # lite-mode path
        cols = [c.lower() for c in df.columns]
        i_frame = cols.index("frame"); i_name = cols.index("name")
        i_x, i_y, i_z = cols.index("x"), cols.index("y"), cols.index("z")
        try:
            i_t = cols.index("time_s")
        except ValueError:
            try:
                i_t = cols.index("time")
            except ValueError:
                i_t = None
        arr = df.to_numpy()
        frames = sorted({int(r[i_frame]) for r in arr})
        names = sorted({str(r[i_name]) for r in arr})
        F, M = len(frames), len(names)
        times = np.zeros(F) if i_t is None else np.zeros(F)
        XYZ = np.zeros((F, M, 3), float)
        f_to_idx = {f: i for i, f in enumerate(frames)}
        n_to_idx = {n: i for i, n in enumerate(names)}
        for r in arr:
            fi = f_to_idx[int(r[i_frame])]
            mi = n_to_idx[str(r[i_name])]
            XYZ[fi, mi, 0] = float(r[i_x]); XYZ[fi, mi, 1] = float(r[i_y]); XYZ[fi, mi, 2] = float(r[i_z])
            if i_t is not None:
                times[fi] = float(r[i_t])
        if i_t is None:
            times = np.arange(F, dtype=float)
        return XYZ, names, times

def _extract_from_wide(df):
    if pd is not None:
        d = df.rename(columns={c: c.lower() for c in df.columns})
        if "time" in d.columns and "time_s" not in d.columns:
            d = d.rename(columns={"time": "time_s"})
        if "frame" not in d.columns:
            d.insert(0, "frame", np.arange(len(d), dtype=int))
        pat = re.compile(r"(.+)_([xyz])$")
        bases = {}
        for c in d.columns:
            m = pat.match(c)
            if m:
                bases.setdefault(m.group(1), set()).add(m.group(2))
        names = sorted([b for b, axes in bases.items() if {"x","y","z"}.issubset(axes)])
        F, M = len(d), len(names)
        times = d["time_s"].to_numpy() if "time_s" in d.columns else np.arange(F, dtype=float)
        XYZ = np.zeros((F, M, 3), float)
        for i, n in enumerate(names):
            XYZ[:, i, 0] = d[f"{n}_x"].to_numpy()
            XYZ[:, i, 1] = d[f"{n}_y"].to_numpy()
            XYZ[:, i, 2] = d[f"{n}_z"].to_numpy()
        return XYZ, names, times
    else:
        cols = [c.lower() for c in df.columns]
        try: i_t = cols.index("time_s")
        except ValueError:
            try: i_t = cols.index("time")
            except ValueError: i_t = None
        pat = re.compile(r"(.+)_([xyz])$")
        bases = {}
        for c in df.columns:
            m = pat.match(c.lower())
            if m:
                bases.setdefault(m.group(1), set()).add(m.group(2))
        names = sorted([b for b, axes in bases.items() if {"x","y","z"}.issubset(axes)])
        mat = np.array(df.to_numpy(), dtype=object)
        F, M = mat.shape[0], len(names)
        times = np.arange(F, dtype=float) if i_t is None else np.array([float(x) for x in mat[:, i_t]])
        XYZ = np.zeros((F, M, 3), float)
        for i, n in enumerate(names):
            ix = df.columns.index(f"{n}_x"); iy = df.columns.index(f"{n}_y"); iz = df.columns.index(f"{n}_z")
            XYZ[:, i, 0] = [float(v) for v in mat[:, ix]]
            XYZ[:, i, 1] = [float(v) for v in mat[:, iy]]
            XYZ[:, i, 2] = [float(v) for v in mat[:, iz]]
        return XYZ, names, times

def load_markers_any(path):
    df = _read_table(path)
    fmt = _detect_format(df)
    if fmt == "long":
        return _extract_from_long(df)
    elif fmt == "wide":
        return _extract_from_wide(df)
    else:
        raise ValueError("Unrecognized file format. Expected LONG (frame,time_s,name,x,y,z) "
                         "or WIDE (frame,time_s,Name_x,Name_y,Name_z, ...).")

def infer_fps(times, default=30.0):
    if times is None or len(times) < 2:
        return default
    dt = np.median(np.diff(times))
    return default if dt <= 0 else 1.0 / dt

def _pelvis_guess(markers, names):
    # Try a good root guess from a pelvis/hip marker if present; else centroid.
    idx = None
    if names:
        wanted = ["pelvis","root","hip","torso"]
        lname = [n.lower() for n in names]
        for w in wanted:
            if w in lname:
                idx = lname.index(w); break
    if idx is not None and 0 <= idx < markers.shape[0]:
        return markers[idx]
    return markers.mean(axis=0)

def fit_from_markers(
    xyz_all, names, times,
    *,
    fps=None,
    geom="segment",           # your recorded markers are typically joint-like → use 'segment'
    draw_solids=False,
    save_npz="data/output/test.angles.npz",
    save_video=None           # e.g., "data/output/test_fit.mp4" or ".gif"
):
    """
    xyz_all: [F, K, 3] markers, names: [K] list, times: [F]
    Saves per-frame joint angles (48,) in radians and root translations (3,) to NPZ.
    Optionally renders a video overlay (fitted skeleton + markers).
    """
    F, K, _ = xyz_all.shape
    print(f"[markers] frames={F}, markers={K}")

    # playback rate (only for video)
    if fps is None: fps = infer_fps(times, default=30.0)

    # geometry
    bone_radii = default_bone_radii() if draw_solids else None

    # joint limits & active DoFs
    lower_lim, upper_lim = get_default_joint_limits()
    # If you want strict hinge elbows/knees, uncomment:
    # lower_lim, upper_lim = enforce_pure_hinges_in_limits(lower_lim, upper_lim)
    active_idx = make_active_dof_indices_human_like_hinges()

    # --- Stage A: bone-length calibration on the first frame (optional but helpful) ---
    theta_guess = get_default_joint_angles()
    root_guess  = _pelvis_guess(xyz_all[0], names)
    print("[fit] Calibrating bone lengths on first frame...")
    theta_cal, bl_cal, root_cal, *_ = lm_fit_markers_to_bones(
        BONE_LENGTHS, theta_guess, xyz_all[0], marker_bones=None,
        opt_joint_indices_list=[active_idx],
        use_segment=True,
        optimize_bones=True, optimize_root=True, root_init=root_guess,
        max_iters=30, tolerance=1e-3,
        angle_delta=8e-4, length_delta=8e-4, root_delta=1e-3,
        lm_lambda0=5e-3, lm_lambda_factor=2.0,
        angle_step_clip=np.deg2rad(10.0),
        length_step_clip=0.015, root_step_clip=0.05,
        angle_reg=1.0, bone_reg=5.0, root_reg=0.5,
        marker_weights=np.ones(K),
        joint_limits=(lower_lim, upper_lim),
        verbose=False,
        auto_assign_bones=True, assign_topk=1,
        assign_soft_sigma_factor=0.10,
        assign_enable_gate=True, assign_distance_gate_abs=None, assign_distance_gate_factor=0.8,
        assign_enable_hysteresis=True, assign_hysteresis_margin=0.08,
        assign_enable_temporal_smoothing=False,
        assign_semantic_priors=None,
        strategy="lm+linesearch", line_search_scales=(1.0, 0.5, 0.25, 0.1),
        allow_trial_reassign=False,
        geom=geom, bone_radii=bone_radii,
        marker_batch_size=min(200, K), reassign_every=3, fast_vectorized=True, rng_seed=0
    )
    bl_est = bl_cal.copy()
    theta_prev = theta_cal.copy()
    root_prev  = root_cal.copy()

    # --- Stage B: track every frame ---
    thetas = np.zeros((F, 48), float)
    roots  = np.zeros((F, 3), float)
    rmse   = np.zeros(F, float)

    for fidx in range(F):
        markers_t = xyz_all[fidx]
        # warm-start previous solution
        theta_fit, bl_fit, root_fit, *_ = lm_fit_markers_to_bones(
            bl_est, theta_prev, markers_t, marker_bones=None,
            opt_joint_indices_list=[active_idx],
            use_segment=True,
            optimize_bones=False, optimize_root=True, root_init=root_prev,
            max_iters=20, tolerance=8e-4,
            angle_delta=8e-4, length_delta=8e-4, root_delta=1e-3,
            lm_lambda0=5e-3, lm_lambda_factor=2.0,
            angle_step_clip=np.deg2rad(10.0),
            length_step_clip=0.015, root_step_clip=0.05,
            angle_reg=1.0, bone_reg=5.0, root_reg=0.5,
            marker_weights=np.ones(K),
            joint_limits=(lower_lim, upper_lim),
            verbose=False,
            auto_assign_bones=True, assign_topk=1,
            assign_soft_sigma_factor=0.10,
            assign_enable_gate=True, assign_distance_gate_abs=None, assign_distance_gate_factor=1.0,
            assign_enable_hysteresis=True, assign_hysteresis_margin=0.08,
            assign_enable_temporal_smoothing=False,
            assign_semantic_priors=None,
            strategy="lm+linesearch", line_search_scales=(1.0, 0.5, 0.25, 0.1),
            allow_trial_reassign=False,
            geom=geom, bone_radii=bone_radii,
            marker_batch_size=min(200, K), reassign_every=3, fast_vectorized=True, rng_seed=1+fidx
        )

        thetas[fidx] = theta_fit
        roots[fidx]  = root_fit
        theta_prev, root_prev = theta_fit, root_fit

        # quick RMSE
        jp_fit, _ = get_joint_positions_and_orientations(bl_est, theta_fit, root_pos=root_fit)
        corr = robust_assign_markers(
            markers_t, jp_fit, BONES_IDX, prev_state=None,
            bone_lengths=bl_est, topk=1, soft_sigma_factor=0.1,
            distance_gate_abs=None, distance_gate_factor=1.0, enable_gate=True,
            hysteresis_margin=0.10, enable_hysteresis=True,
            temporal_smoothing=0.0, enable_temporal_smoothing=False,
            semantic_priors=None,
            geom=geom, bone_radii=bone_radii
        )
        rs = build_residual_stack_hard_geom(jp_fit, markers_t, corr['hard'], geom="segment", bone_radii=None).reshape(-1,3)
        rmse[fidx] = np.sqrt(np.mean(np.sum(rs*rs, axis=1)))

    # --- Save outputs ---
    out_path = save_npz or "test.angles.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        joint_angles_rad=thetas,
        root_translation=roots,
        bone_lengths=np.array([bl_est[k] for k in BONE_LENGTHS.keys()], dtype=float),
        bone_length_keys=np.array(list(BONE_LENGTHS.keys())),
        marker_names=np.array(names),
        times=times,
        rmse=rmse
    )
    print(f"[fit] Saved joint angles + roots to: {out_path}")
    print(f"[fit] RMSE: mean={rmse.mean():.4f} m, median={np.median(rmse):.4f} m")

    # --- Optional: render an overlay video ---
    if save_video:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # bounds from data for clean camera
        mins = xyz_all.reshape(-1,3).min(axis=0)
        maxs = xyz_all.reshape(-1,3).max(axis=0)
        center = 0.5*(mins+maxs); span = float(np.max(maxs-mins)); span = 1.0 if span<=0 else span
        pad = 0.1*span

        def frame(i):
            ax.clear()
            # fitted skeleton
            jp, _ = get_joint_positions_and_orientations(bl_est, thetas[i], root_pos=roots[i])
            plot_skeleton(ax, jp, [None]*len(jp),
                          markers=None, marker_bones=None, show_axes=False,
                          title=f'Frame {i+1}/{F}  RMSE={rmse[i]:.3f} m',
                          draw_solids=draw_solids, bone_radii=bone_radii,
                          clear=False, joint_color='k', wire_color='k', wire_alpha=1.0)
            # markers
            mk = xyz_all[i]
            ax.scatter(mk[:,0], mk[:,1], mk[:,2], marker='x', s=28, color='C1', alpha=0.9)
            ax.set_xlim(center[0]-0.5*span-pad, center[0]+0.5*span+pad)
            ax.set_ylim(center[1]-0.5*span-pad, center[1]+0.5*span+pad)
            ax.set_zlim(center[2]-0.5*span-pad, center[2]+0.5*span+pad)
            set_axes_equal(ax)

        ani = FuncAnimation(fig, frame, frames=F, interval=1000.0/fps, repeat=False)
        base, ext = os.path.splitext(save_video)
        if ext.lower() == ".mp4":
            save_animation(ani, base, fps=int(round(fps)), dpi=150)
        else:
            # gif path: pass "xxx.gif"; save_animation() expects base wo/ ext,
            # so do a simple GIF save here
            from matplotlib.animation import PillowWriter
            ani.save(save_video, writer=PillowWriter(fps=int(round(fps))), dpi=150)
            print(f"Saved {save_video}")
        plt.close(fig)


def _read_table_debug(path: str):
    if pd is not None:
        return pd.read_csv(path, sep=None, engine="python")
    # tiny fallback CSV reader
    import csv
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    data = []
    for r in rows[1:]:
        row = []
        for x in r:
            try:
                row.append(float(x))
            except Exception:
                row.append(x)
        data.append(row)
    class _Lite:
        def __init__(self, header, data):
            self.columns = header
            self._data = data
        def __getitem__(self, k):
            i = self.columns.index(k)
            return np.array([row[i] for row in self._data], dtype=object)
        def to_numpy(self): return np.array(self._data, dtype=object)
    return _Lite(header, data)

def _detect_format_debug(df):
    cols = [str(c).lower() for c in (df.columns if pd is not None else df.columns)]
    if {"frame","name","x","y","z"}.issubset(set(cols)): return "long"
    pat = re.compile(r".+_([xyz])$", re.IGNORECASE)
    has_xyz = any(pat.match(c) for c in (df.columns if pd is not None else df.columns))
    return "wide" if has_xyz else "unknown"

def load_markers_any_debug(path):
    df = _read_table_debug(path)
    fmt = _detect_format_debug(df)

    if fmt == "long":
        if pd is not None:
            d = df.rename(columns={c: c.lower() for c in df.columns})
            names = sorted(d["name"].unique().tolist())
            frames = np.sort(d["frame"].unique())
            # take frame 0 only
            f0 = frames[0]
            sub = d[d["frame"] == f0].drop_duplicates(subset=["name"], keep="last")
            sub = sub.set_index("name")
            K = len(names)
            XYZ = np.zeros((K, 3), float)
            for i, n in enumerate(names):
                if n in sub.index:
                    XYZ[i] = [float(sub.loc[n,"x"]), float(sub.loc[n,"y"]), float(sub.loc[n,"z"])]
            time0 = float(sub["time_s"].iloc[0]) if "time_s" in sub.columns else 0.0
            return XYZ, names, time0
        else:
            cols = [c.lower() for c in df.columns]
            iF, iN = cols.index("frame"), cols.index("name")
            iX, iY, iZ = cols.index("x"), cols.index("y"), cols.index("z")
            try: iT = cols.index("time_s")
            except ValueError:
                try: iT = cols.index("time")
                except ValueError: iT = None
            arr = df.to_numpy()
            frames = sorted({int(r[iF]) for r in arr})
            f0 = frames[0]
            rows0 = [r for r in arr if int(r[iF]) == f0]
            names = []
            seen = set()
            for r in rows0:
                n = str(r[iN])
                if n not in seen:
                    seen.add(n); names.append(n)
            K = len(names)
            XYZ = np.zeros((K,3), float)
            name_to_i = {n:i for i,n in enumerate(names)}
            t0 = None
            for r in rows0:
                n = str(r[iN]); i = name_to_i[n]
                XYZ[i] = [float(r[iX]), float(r[iY]), float(r[iZ])]
                if iT is not None: t0 = float(r[iT])
            return XYZ, names, (0.0 if t0 is None else t0)

    elif fmt == "wide":
        # take the first row only
        if pd is not None:
            d = df.rename(columns={c: c.lower() for c in df.columns})
            pat = re.compile(r"(.+)_([xyz])$")
            bases = {}
            for c in d.columns:
                m = pat.match(c)
                if m: bases.setdefault(m.group(1), set()).add(m.group(2))
            names = sorted([b for b, axes in bases.items() if {"x","y","z"}.issubset(axes)])
            row0 = d.iloc[0]
            XYZ = np.zeros((len(names),3), float)
            for i,n in enumerate(names):
                XYZ[i] = [row0[f"{n}_x"], row0[f"{n}_y"], row0[f"{n}_z"]]
            t0 = float(row0["time_s"]) if "time_s" in d.columns else 0.0
            return XYZ, names, t0
        else:
            cols = [c.lower() for c in df.columns]
            pat = re.compile(r"(.+)_([xyz])$")
            bases = {}
            for c in df.columns:
                m = pat.match(c.lower())
                if m: bases.setdefault(m.group(1), set()).add(m.group(2))
            names = sorted([b for b, axes in bases.items() if {"x","y","z"}.issubset(axes)])
            arr = df.to_numpy()
            row0 = arr[0]
            XYZ = np.zeros((len(names),3), float)
            for i,n in enumerate(names):
                ix = df.columns.index(f"{n}_x"); iy = df.columns.index(f"{n}_y"); iz = df.columns.index(f"{n}_z")
                XYZ[i] = [float(row0[ix]), float(row0[iy]), float(row0[iz])]
            try:
                it = cols.index("time_s"); t0 = float(row0[it])
            except ValueError:
                t0 = 0.0
            return XYZ, names, t0

    raise ValueError("Unrecognized markers file format. Expect LONG (frame,time_s,name,x,y,z) or WIDE (.._x,.._y,.._z).")

# ---- small helpers -----------------------------------------------------------
def _root_guess_from_markers(markers: np.ndarray, names):
    # Prefer a pelvis/root/hip marker if present; else centroid
    idx = None
    lname = [n.lower() for n in names]
    for key in ("pelvis","root","hip","torso"):
        if key in lname:
            idx = lname.index(key); break
    return (markers[idx] if idx is not None else markers.mean(axis=0))

def _closest_point_geom(marker, pa, pb, geom, radius):
    if geom == "segment":
        return _closest_point_on_segment_pointwise(marker, pa, pb)
    elif geom == "capsule":
        return closest_point_on_capsule_surface(marker, pa, pb, radius)
    elif geom == "cylinder":
        return closest_point_on_capped_cylinder_surface(marker, pa, pb, radius)
    else:
        raise ValueError("geom must be 'segment' | 'capsule' | 'cylinder'")

# ---- canon helper for name->index mapping ------------------
def _canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

_CANON_TO_JIDX = {_canon(n): i for n, i in JOINT_IDX.items()}

def _name_to_idx(s: str) -> int:
    key = _canon(s)
    if key not in _CANON_TO_JIDX:
        raise KeyError(f"Unknown joint name '{s}' (canon='{key}')")
    return _CANON_TO_JIDX[key]

# ---- file loader: markers (frame 0) + optional marker_bones ----------
def _read_table_any(path: str):
    if pd is not None:
        return pd.read_csv(path, sep=None, engine="python")
    import csv
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    data = []
    for r in rows[1:]:
        row = []
        for x in r:
            try:
                row.append(float(x))
            except Exception:
                row.append(x)
        data.append(row)
    class _Lite:
        def __init__(self, header, data):
            self.columns = header
            self._data = data
        def __getitem__(self, k):
            i = self.columns.index(k)
            return np.array([row[i] for row in self._data], dtype=object)
        def to_numpy(self): return np.array(self._data, dtype=object)
    return _Lite(header, data)

def _detect_wide_vs_long(df) -> str:
    cols = [str(c).lower() for c in df.columns]
    if {"frame", "name", "x", "y", "z"}.issubset(set(cols)):
        return "long"
    has_xyz = any(re.match(r".+_([xyz])$", c) for c in cols)
    return "wide" if has_xyz else "unknown"

def _try_sidecar_json(path_txt: str):
    base, ext = os.path.splitext(path_txt)
    cand = base + ".marker_bones.json"
    if os.path.isfile(cand):
        with open(cand, "r") as f:
            blob = json.load(f)
        # accepted shapes:
        # - {"marker_bones":[[ja,jb],...], "names":[...]}
        # - [[ja,jb], ...]
        if isinstance(blob, dict) and "marker_bones" in blob:
            mb = [(int(a), int(b)) for (a, b) in blob["marker_bones"]]
            names = blob.get("names", None)
            return mb, names
        elif isinstance(blob, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in blob):
            return [(int(a), int(b)) for (a, b) in blob], None
    return None, None

def load_first_frame_with_bones(path: str):
    """
    Returns:
      markers0: (K,3) float
      names:    list[str] (length K) or None
      marker_bones: list[(ja,jb)] (length K) or None if missing
      t0:       time of frame 0 (float, 0.0 if absent)
    """
    df = _read_table_any(path)
    if pd is not None:
        df = df.rename(columns={c: str(c).lower() for c in df.columns})
    cols = [str(c).lower() for c in df.columns]
    fmt = _detect_wide_vs_long(df)

    names = None
    marker_bones = None
    t0 = 0.0

    if fmt == "long":
        # frame 0 slice
        frames = np.unique(np.array(df["frame"], dtype=int))
        f0 = int(frames[0])
        if pd is not None:
            sub = df[df["frame"] == f0]
        else:
            arr = df.to_numpy()
            sub_rows = [r for r in arr if int(r[cols.index("frame")]) == f0]
            # rebuild minimal "df-like" for uniform downstream:
            if pd is not None:
                raise RuntimeError("Pandas-less long loader expects pandas; install pandas or use WIDE/JSON sidecar.")
            sub = None  # not used further when pandas is missing

        # names
        if pd is not None:
            names = sub["name"].astype(str).tolist()
        # coords
        XYZ = np.stack([sub["x"].astype(float).to_numpy(),
                        sub["y"].astype(float).to_numpy(),
                        sub["z"].astype(float).to_numpy()], axis=1)

        # time
        if "time_s" in cols:
            t0 = float(sub["time_s"].iloc[0])
        elif "time" in cols:
            t0 = float(sub["time"].iloc[0])

        if marker_bones is None and names is not None:
            mb_guess = build_marker_bones_from_amass_names(names)
            use_auto = (mb_fixed is None)
            if mb_guess is not None:
                marker_bones = mb_guess
                print("[debug] inferred marker_bones from AMASS names.")
                
                
        # marker_bones from columns?
        if "ja" in cols and "jb" in cols:
            ja = sub["ja"].astype(int).to_numpy()
            jb = sub["jb"].astype(int).to_numpy()
            marker_bones = [(int(a), int(b)) for a, b in zip(ja, jb)]
        elif "bone" in cols:
            mb = []
            for s in sub["bone"].astype(str).tolist():
                s = s.strip()
                if re.match(r"^\d+\s*[-:]\s*\d+$", s):
                    a, b = re.split(r"[-:]", s)
                    mb.append((int(a), int(b)))
                else:
                    # name-name, e.g. "SpineTop-RightShoulder"
                    parts = re.split(r"[-:/\\]", s)
                    if len(parts) != 2:
                        raise ValueError(f"Can't parse bone spec: '{s}'")
                    a = _name_to_idx(parts[0])
                    b = _name_to_idx(parts[1])
                    mb.append((a, b))
            marker_bones = mb
        elif ("ja_name" in cols) and ("jb_name" in cols):
            ja = [_name_to_idx(s) for s in sub["ja_name"].astype(str).tolist()]
            jb = [_name_to_idx(s) for s in sub["jb_name"].astype(str).tolist()]
            marker_bones = list(zip(ja, jb))
        else:
            # try sidecar json
            marker_bones, _ = _try_sidecar_json(path)

        return XYZ, names, marker_bones, t0

    elif fmt == "wide":
        # first row
        if pd is None:
            raise RuntimeError("WIDE without pandas not supported in this loader.")
        row0 = df.iloc[0]
        # collect names by _x/_y/_z triplets
        bases = {}
        for c in df.columns:
            m = re.match(r"(.+)_([xyz])$", str(c))
            if m:
                bases.setdefault(m.group(1), set()).add(m.group(2))
        names = sorted([b for b, axes in bases.items() if {"x","y","z"}.issubset(axes)])
        XYZ = np.zeros((len(names), 3), float)
        for i, n in enumerate(names):
            XYZ[i] = [float(row0[f"{n}_x"]), float(row0[f"{n}_y"]), float(row0[f"{n}_z"])]
        if "time_s" in df.columns:
            t0 = float(row0["time_s"])
        # mapping must come from sidecar JSON (or from extra *_ja/*_jb columns if you added them)
        # try *_ja/*_jb
        ja_cols = [f"{n}_ja" for n in names]
        jb_cols = [f"{n}_jb" for n in names]
        if all(c in df.columns for c in ja_cols + jb_cols):
            marker_bones = [(int(row0[f"{n}_ja"]), int(row0[f"{n}_jb"])) for n in names]
        else:
            marker_bones, _ = _try_sidecar_json(path)
        return XYZ, names, marker_bones, t0

    else:
        raise ValueError("Unrecognized markers file format. Expect LONG (frame,time_s,name,x,y,z[,ja,jb|bone]) or WIDE (..._x,_y,_z).")

# ---- closest-point helpers (reuse your geometry) ----------------------------
def _closest_point_geom(marker, pa, pb, geom, radius):
    if geom == "segment":
        return _closest_point_on_segment_pointwise(marker, pa, pb)
    elif geom == "capsule":
        return closest_point_on_capsule_surface(marker, pa, pb, radius)
    elif geom == "cylinder":
        return closest_point_on_capped_cylinder_surface(marker, pa, pb, radius)
    else:
        raise ValueError("geom must be 'segment' | 'capsule' | 'cylinder'")

# ---- main debug (frame 0, fixed correspondences) ----------------------------
def debug_first_frame_known_bones(markers_path: str,
                                  geom: str = "segment",
                                  draw_solids: bool = False,
                                  annotate_max: int = 20,
                                  save_png: str | None = None):
    # 1) load first frame + mapping
    mk0, names, marker_bones, t0 = load_first_frame_with_bones(markers_path)
    K = mk0.shape[0]
    if marker_bones is None or len(marker_bones) != K:
        print("[debug] WARNING: marker_bones not found or length mismatch; falling back to auto-assign.")
        use_auto = True
    else:
        use_auto = False
        print(f"[debug] using provided marker_bones ({K} entries) from AMASS/file.")

    # 2) radii (only used for cylinder/capsule rendering or closest-point)
    bone_radii = default_bone_radii() if (draw_solids or geom in ("cylinder","capsule")) else None

    # 3) initial guess
    theta0 = get_default_joint_angles()
    root0  = mk0.mean(axis=0)  # or pick pelvis marker if you have one by name

    # 4) fit (bones fixed to mapping if available)
    lower, upper = get_default_joint_limits()
    active_idx = make_active_dof_indices_human_like_hinges()

    theta_fit, bl_fit, root_fit, *_ = lm_fit_markers_to_bones(
        BONE_LENGTHS, theta0, mk0,
        marker_bones=None if use_auto else marker_bones,
        opt_joint_indices_list=[active_idx],
        use_segment=True,
        optimize_bones=True, optimize_root=True, root_init=root0,
        max_iters=50, tolerance=8e-4,
        angle_delta=8e-4, length_delta=8e-4, root_delta=1e-3,
        lm_lambda0=5e-3, lm_lambda_factor=2.0,
        angle_step_clip=np.deg2rad(10.0),
        length_step_clip=0.015, root_step_clip=0.05,
        angle_reg=1.0, bone_reg=5.0, root_reg=0.5,
        marker_weights=np.ones(K),
        joint_limits=(lower, upper),
        verbose=False,
        auto_assign_bones=use_auto, assign_topk=1,
        assign_soft_sigma_factor=0.10,
        assign_enable_gate=True, assign_distance_gate_abs=None, assign_distance_gate_factor=1.0,
        assign_enable_hysteresis=True, assign_hysteresis_margin=0.08,
        assign_enable_temporal_smoothing=False,
        assign_semantic_priors=None,
        strategy="lm+linesearch", line_search_scales=(1.0, 0.5, 0.25, 0.1),
        allow_trial_reassign=False,
        geom=geom, bone_radii=bone_radii,
        marker_batch_size=min(200, K), reassign_every=3, fast_vectorized=True, rng_seed=0
    )

    # 5) FK and residuals (using fixed mapping if provided)
    jp_fit, _ = get_joint_positions_and_orientations(bl_fit, theta_fit, root_pos=root_fit)

    if use_auto:
        # compute correspondences with your existing function
        corr = robust_assign_markers(
            mk0, jp_fit, BONES_IDX, prev_state=None,
            bone_lengths=bl_fit, topk=1, soft_sigma_factor=0.1,
            distance_gate_abs=None, distance_gate_factor=1.0, enable_gate=True,
            hysteresis_margin=0.10, enable_hysteresis=True,
            temporal_smoothing=0.0, enable_temporal_smoothing=False,
            semantic_priors=None,
            geom=geom, bone_radii=bone_radii
        )
        hard_pairs = corr['hard']
    else:
        hard_pairs = marker_bones

    dists = np.zeros(K)
    closest_pts = np.zeros_like(mk0)
    for k, (ja, jb) in enumerate(hard_pairs):
        R = 0.0
        if bone_radii is not None:
            bi = BONE_PAIR_TO_INDEX[(ja, jb)]
            R = float(bone_radii[bi])
        cp = _closest_point_geom(mk0[k], jp_fit[ja], jp_fit[jb], geom=geom, radius=R)
        closest_pts[k] = cp
        dists[k] = np.linalg.norm(mk0[k] - cp)

    rmse = float(np.sqrt(np.mean(dists**2)))
    print(f"[debug] frame0 @ t={t0:.3f}s  RMSE={rmse:.4f}m  mean={dists.mean():.4f}  max={dists.max():.4f}")

    # 6) visualize
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    plot_skeleton(ax, jp_fit, [None]*len(jp_fit),
                  markers=None, marker_bones=None, show_axes=False,
                  title=f"First-frame fit (known correspondences) | RMSE={rmse:.3f} m | geom={geom}",
                  draw_solids=(bone_radii is not None), bone_radii=bone_radii,
                  clear=True, joint_color='k', wire_color='k', wire_alpha=1.0)
    ax.scatter(mk0[:,0], mk0[:,1], mk0[:,2], marker='x', s=36, color='C1', alpha=0.95, label='markers')

    # residual lines (color by distance)
    scale = np.percentile(dists, 95) if np.any(dists>0) else 1.0
    cmap = cm.get_cmap('viridis')
    for k in range(K):
        pa = mk0[k]; pb = closest_pts[k]
        c = cmap(min(1.0, float(dists[k]/(scale + 1e-12))))
        ax.plot([pa[0], pb[0]],[pa[1], pb[1]],[pa[2], pb[2]], linewidth=2.0, color=c, alpha=0.9)

    # annotate worst N markers
    if names is not None:
        order = np.argsort(-dists)[:min(annotate_max, K)]
        for k in order:
            ja, jb = hard_pairs[k]
            lbl = f"{names[k]} | {JOINT_NAMES[ja]}–{JOINT_NAMES[jb]}: {dists[k]:.3f}m"
            ax.text(mk0[k,0], mk0[k,1], mk0[k,2], lbl, fontsize=8, color='black', ha='left', va='bottom')

    ax.legend(loc='upper left')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    set_axes_equal(ax)

    if save_png:
        os.makedirs(os.path.dirname(save_png), exist_ok=True)
        plt.savefig(save_png, dpi=200)
        print(f"[debug] saved figure to {save_png}")
    plt.show()

# ---- CLI entry for just this debug ------------------------------------------
def main_debug0_known_bones():
    parser = argparse.ArgumentParser(description="First-frame marker debug with known marker_bones")
    parser.add_argument("input", type=str, nargs="?", default="data/output/test.markers.txt",
                        help="Markers file (LONG: frame,time_s,name,x,y,z[,ja,jb|bone]) or WIDE (..._x,_y,_z)")
    parser.add_argument("--geom", type=str, default="segment", choices=["segment","capsule","cylinder"],
                        help="Residual geometry for correspondences")
    parser.add_argument("--solids", action="store_true", help="Draw cylinders/capsules (uses default radii)")
    parser.add_argument("--annot", type=int, default=20, help="Annotate top-N largest residuals")
    parser.add_argument("--save", type=str, default=None, help="Optional PNG path")
    args = parser.parse_args()
    debug_first_frame_known_bones(args.input, geom=args.geom, draw_solids=args.solids,
                                  annotate_max=args.annot, save_png=args.save)

def debug_first_frame(markers_path: str,
                      geom: str = "segment",
                      draw_solids: bool = False,
                      annotate_max: int = 20,
                      save_png: str | None = None):
    """
    Loads markers, fits ONLY frame 0, and visualizes correspondences & residuals.
    """
    # 1) load first frame of markers
    mk0, names, t0 = load_markers_any_debug(markers_path)
    K = mk0.shape[0]
    print(f"[debug] loaded {markers_path} | frame0 K={K} markers at t={t0:.3f}s")

    # 2) geometry/radii
    bone_radii = default_bone_radii() if draw_solids or (geom in ("cylinder","capsule")) else None

    # 3) initial guess (angles = zeros; root from pelvis/centroid)
    theta0 = get_default_joint_angles()
    root0  = _root_guess_from_markers(mk0, names)

    # 4) calibrate bone lengths (only on frame 0) + fit angles/root
    lower, upper = get_default_joint_limits()
    # (optional) enforce pure hinge elbows/knees
    # lower, upper = enforce_pure_hinges_in_limits(lower, upper)

    active_idx = make_active_dof_indices_human_like_hinges()

    print("[debug] solving first frame...")
    theta_fit, bl_fit, root_fit, *_ = lm_fit_markers_to_bones(
        BONE_LENGTHS, theta0, mk0, marker_bones=None,
        opt_joint_indices_list=[active_idx],
        use_segment=True,
        optimize_bones=True, optimize_root=True, root_init=root0,
        max_iters=50, tolerance=8e-4,
        angle_delta=8e-4, length_delta=8e-4, root_delta=1e-3,
        lm_lambda0=5e-3, lm_lambda_factor=2.0,
        angle_step_clip=np.deg2rad(10.0),
        length_step_clip=0.015, root_step_clip=0.05,
        angle_reg=1.0, bone_reg=5.0, root_reg=0.5,
        marker_weights=np.ones(K),
        joint_limits=(lower, upper),
        verbose=False,
        auto_assign_bones=True, assign_topk=1,
        assign_soft_sigma_factor=0.10,
        assign_enable_gate=True, assign_distance_gate_abs=None, assign_distance_gate_factor=1.0,
        assign_enable_hysteresis=True, assign_hysteresis_margin=0.08,
        assign_enable_temporal_smoothing=False,
        assign_semantic_priors=None,
        strategy="lm+linesearch", line_search_scales=(1.0, 0.5, 0.25, 0.1),
        allow_trial_reassign=False,
        geom=geom, bone_radii=bone_radii,
        marker_batch_size=min(200, K), reassign_every=3, fast_vectorized=True, rng_seed=0
    )

    # 5) forward kinematics for fitted pose
    jp_fit, _ = get_joint_positions_and_orientations(bl_fit, theta_fit, root_pos=root_fit)

    # 6) compute correspondences and residuals for display
    corr = robust_assign_markers(
        mk0, jp_fit, BONES_IDX, prev_state=None,
        bone_lengths=bl_fit, topk=1, soft_sigma_factor=0.1,
        distance_gate_abs=None, distance_gate_factor=1.0, enable_gate=True,
        hysteresis_margin=0.10, enable_hysteresis=True,
        temporal_smoothing=0.0, enable_temporal_smoothing=False,
        semantic_priors=None,
        geom=geom, bone_radii=bone_radii
    )

    # residuals
    dists = np.zeros(K)
    closest_pts = np.zeros_like(mk0)
    for k, (ja, jb) in enumerate(corr['hard']):
        R = 0.0
        if bone_radii is not None:
            bi = BONE_PAIR_TO_INDEX[(ja, jb)]
            R = float(bone_radii[bi])
        cp = _closest_point_geom(mk0[k], jp_fit[ja], jp_fit[jb], geom=geom, radius=R)
        closest_pts[k] = cp
        dists[k] = np.linalg.norm(mk0[k] - cp)

    rmse = float(np.sqrt(np.mean(dists**2)))
    print(f"[debug] frame0 RMSE = {rmse:.4f} m  |  mean={dists.mean():.4f}  max={dists.max():.4f}")

    # 7) visualize
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    # skeleton
    plot_skeleton(ax, jp_fit, [None]*len(jp_fit),
                  markers=None, marker_bones=None, show_axes=False,
                  title=f"First-frame fit | RMSE={rmse:.3f} m | geom={geom}",
                  draw_solids=(bone_radii is not None), bone_radii=bone_radii,
                  clear=True, joint_color='k', wire_color='k', wire_alpha=1.0)
    # markers
    ax.scatter(mk0[:,0], mk0[:,1], mk0[:,2], marker='x', s=36, color='C1', alpha=0.95, label='markers')

    # residual lines colored by distance
    scale = np.percentile(dists, 95) if np.any(dists>0) else 1.0
    cmap = cm.get_cmap('viridis')
    for k in range(K):
        pa = mk0[k]; pb = closest_pts[k]
        c = cmap(min(1.0, float(dists[k]/(scale + 1e-12))))
        ax.plot([pa[0], pb[0]],
                [pa[1], pb[1]],
                [pa[2], pb[2]],
                linewidth=2.0, color=c, alpha=0.9)

    # annotate a few largest errors
    order = np.argsort(-dists)[:min(annotate_max, K)]
    for k in order:
        ja, jb = corr['hard'][k]
        ax.text(mk0[k,0], mk0[k,1], mk0[k,2],
                f"{names[k] if k < len(names) else k}\n({JOINT_NAMES[ja]}–{JOINT_NAMES[jb]}): {dists[k]:.3f}m",
                fontsize=8, color='black', ha='left', va='bottom')

    ax.legend(loc='upper left')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    set_axes_equal(ax)

    if save_png:
        os.makedirs(os.path.dirname(save_png), exist_ok=True)
        plt.savefig(save_png, dpi=200)
        print(f"[debug] saved figure to {save_png}")

    plt.show()

# ---- CLI hook: run only first-frame debug and exit --------------------------
def main_debug0():
    parser = argparse.ArgumentParser(description="First-frame marker debug visualizer")
    parser.add_argument("input", type=str, nargs="?", default="data/output/test.markers.txt",
                        help="Markers file (long or wide format)")
    parser.add_argument("--geom", type=str, default="segment", choices=["segment","capsule","cylinder"],
                        help="Residual geometry for correspondences")
    parser.add_argument("--solids", action="store_true", help="Draw cylinders (uses default radii)")
    parser.add_argument("--annot", type=int, default=20, help="Annotate top-N largest residuals")
    parser.add_argument("--save", type=str, default=None, help="Optional PNG path to save the figure")
    args = parser.parse_args()
    debug_first_frame(args.input, geom=args.geom, draw_solids=args.solids,
                      annotate_max=args.annot, save_png=args.save)

def main_from_markers():
    ap = argparse.ArgumentParser(description="Fit joint angles from marker txt and (optionally) render.")
    ap.add_argument("input", type=str, nargs="?", default="data/output/test.markers.txt",
                    help="Path to markers file (long or wide format).")
    ap.add_argument("--fps", type=float, default=None, help="Override FPS (else inferred from time_s or 30).")
    ap.add_argument("--geom", type=str, default="segment", choices=["segment","cylinder","capsule"],
                    help="Residual geometry; 'segment' is best for joint-like markers.")
    ap.add_argument("--video", type=str, default=None, help="Output video path (.mp4 or .gif).")
    ap.add_argument("--npz", type=str, default="data/output/test.angles.npz", help="Where to save fitted results.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    XYZ, names, times = load_markers_any(args.input)
    fps = args.fps if args.fps is not None else infer_fps(times, default=30.0)

    fit_from_markers(
        XYZ, names, times,
        fps=fps,
        geom=args.geom,
        draw_solids=(args.geom in ("cylinder","capsule")),
        save_npz=args.npz,
        save_video=args.video
    )

if __name__ == "__main__":
    # main_from_markers()
    # main_debug0()
    main_debug0_known_bones()
