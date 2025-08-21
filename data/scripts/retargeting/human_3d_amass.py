import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib as mpl
import time
from collections import defaultdict

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
# AMASS loader & mapping         #
# ============================== #
def load_amass_markers_file(
    path,
    *,
    axis_permutation=(0, 1, 2),   # change if axes need reordering
    axis_flip=(1.0, 1.0, 1.0)     # change to (-1,1,1) etc if a sign flip is needed
):
    """
    Read AMASS-style CSV (data/output/test.markers.txt) with columns:
      frame,time_s,Pelvis_x,Pelvis_y,Pelvis_z, ... ,R_Wrist_x,R_Wrist_y,R_Wrist_z

    Returns:
      times:            (T,) float seconds
      markers_seq:      (T, K, 3) float markers, K=16 in our mapping
      marker_bones:     list of K (ja,jb) bone pairs, aligned with markers order
    """
    # Your skeleton joint indices are already defined above.
    # Map each AMASS joint (base name) to a bone segment (ja, jb) to "pin" it to.
    AMASS_TO_OUR = [
        ("Pelvis",      (PELVIS, SPINE_TOP)),

        ("Chest",       (SPINE_TOP, NECK_TOP)),   # spine_top
        ("Neck",        (SPINE_TOP, NECK_TOP)),   # neck_top @ that segment start
        ("Head",        (NECK_TOP,  HEAD_TOP)),   # head_top

        ("R_Shoulder",  (SPINE_TOP, RIGHT_SHOULDER)),
        ("R_Elbow",     (RIGHT_SHOULDER, RIGHT_ELBOW)),
        ("R_Wrist",     (RIGHT_ELBOW, RIGHT_HAND)),   # AMASS has Wrist; we use it as "hand"

        ("L_Shoulder",  (SPINE_TOP, LEFT_SHOULDER)),
        ("L_Elbow",     (LEFT_SHOULDER, LEFT_ELBOW)),
        ("L_Wrist",     (LEFT_ELBOW, LEFT_HAND)),

        ("R_Hip",       (PELVIS, RIGHT_HIP)),
        ("R_Knee",      (RIGHT_HIP, RIGHT_KNEE)),
        ("R_Ankle",     (RIGHT_KNEE, RIGHT_FOOT)),

        ("L_Hip",       (PELVIS, LEFT_HIP)),
        ("L_Knee",      (LEFT_HIP, LEFT_KNEE)),
        ("L_Ankle",     (LEFT_KNEE, LEFT_FOOT)),
    ]

    # Robust CSV read using numpy (no hard dependency on pandas).
    rec = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None, autostrip=True)
    # Ensure 2D "rows" even if only one line
    if rec.shape == ():
        rec = rec.reshape(1)
    cols = rec.dtype.names

    def _have(base):
        return (f"{base}_x" in cols) and (f"{base}_y" in cols) and (f"{base}_z" in cols)

    missing = [base for base, _ in AMASS_TO_OUR if not _have(base)]
    if missing:
        raise ValueError(f"AMASS file is missing columns for: {missing}")

    T = rec.shape[0]
    K = len(AMASS_TO_OUR)
    markers_seq = np.zeros((T, K, 3), dtype=float)

    perm = np.array(axis_permutation, dtype=int)
    flip = np.array(axis_flip, dtype=float).reshape(1, 3)

    for k, (base, _seg) in enumerate(AMASS_TO_OUR):
        xyz = np.stack([rec[f"{base}_x"], rec[f"{base}_y"], rec[f"{base}_z"]], axis=1)
        # Optional axis remap / flips to match your X-forward, Y-left, Z-up convention
        xyz = xyz[:, perm] * flip
        markers_seq[:, k, :] = xyz

    # Time
    if "time_s" in cols:
        times = np.asarray(rec["time_s"], dtype=float)
    else:
        # Fallback: infer 30 fps
        times = np.arange(T, dtype=float) / 30.0

    marker_bones = [seg for _, seg in AMASS_TO_OUR]
    return times, markers_seq, marker_bones

def load_markers_csv_generic(
    path,
    *,
    axis_permutation=(0, 1, 2),
    axis_flip=(1.0, 1.0, 1.0)
):
    """
    Generic loader: reads any columns that appear as <name>_x, <name>_y, <name>_z.
    Returns:
      times:       (T,) seconds (from 'time_s' if present; else 30 FPS)
      markers_seq: (T, K, 3) float
      names:       list[str] length K (the detected base names, in file order)
    """
    rec = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None, autostrip=True)
    if rec.shape == ():
        rec = rec.reshape(1)
    cols = rec.dtype.names

    # Detect bases in the *file order* (so the sequence is stable)
    bases_in_order = []
    seen = set()
    for c in cols:
        if c.endswith("_x"):
            base = c[:-2]
            if (f"{base}_y" in cols) and (f"{base}_z" in cols) and (base not in seen):
                bases_in_order.append(base)
                seen.add(base)

    T = rec.shape[0]
    K = len(bases_in_order)
    if K == 0:
        raise ValueError("No <name>_{x,y,z} triplets found in the CSV header.")

    perm = np.array(axis_permutation, dtype=int)
    flip = np.array(axis_flip, dtype=float).reshape(1, 3)

    markers_seq = np.zeros((T, K, 3), dtype=float)
    for k, base in enumerate(bases_in_order):
        xyz = np.stack([rec[f"{base}_x"], rec[f"{base}_y"], rec[f"{base}_z"]], axis=1)
        xyz = xyz[:, perm] * flip
        markers_seq[:, k, :] = xyz

    if "time_s" in cols:
        times = np.asarray(rec["time_s"], dtype=float)
    else:
        times = np.arange(T, dtype=float) / 30.0  # default 30 FPS

    return times, markers_seq, bases_in_order

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

# ====================== #
# Demo / Main            #
# ====================== #
if __name__ == "__main__":
    # --- Choose GT geometry for marker generation ---
    GT_GEOM = "cylinder"     # "segment" | "cylinder" | "capsule"
    DRAW_SOLIDS = GT_GEOM in ("cylinder", "capsule")
    ANIMATE_FRAMES = True
    USE_AMASS   = False
    AMASS_FILE  = "data/output/test.markers.txt"   # <-- point to your file
    AMASS_PERM  = (0, 1, 2)    # change if needed, e.g., (2,0,1)
    AMASS_FLIP  = (1.0, 1.0, 1.0)  # change to (-1,1,1) etc if axes are mirrored

    
    # --- walking sequence settings ---
    N_FRAMES = 120
    DT = 1.0 / 30.0
    fps = int(round(1/DT))

    SPEED = 1.0   # m/s forward
    GAIT_F = 1.2  # Hz

    # joint limits & active DoFs
    lower_lim, upper_lim = get_default_joint_limits()
    active_idx = make_active_dof_indices_human_like_hinges()

    # Solver knobs
    angle_delta  = 8e-4
    length_delta = 8e-4
    root_delta   = 1e-3
    angle_step_clip  = np.deg2rad(10.0)
    length_step_clip = 0.015
    root_step_clip   = 0.05
    ASSIGN_TOPK      = 1
    
    # speed knobs for per-frame IK
    BATCH_SZ = 150
    REASSIGN_EVERY = 3
    FAST_VEC = True
    MEAS_NOISE_STD = 0.02  # 2 cm

    # Geometry for residuals
    SOLVER_GEOM      = "cylinder"     # for joint markers, "segment" is appropriate
    DRAW_SOLIDS      = True         # we won’t render capped cylinders by default


    if USE_AMASS:
        # -------------------- Load AMASS -------------------------------------
        times, markers_seq, marker_bones = load_amass_markers_file(
            AMASS_FILE, axis_permutation=AMASS_PERM, axis_flip=AMASS_FLIP
        )
        N_FRAMES = len(times)
        if N_FRAMES == 0:
            raise RuntimeError("No frames in AMASS file.")
        DT  = float(np.median(np.diff(times))) if N_FRAMES > 1 else (1.0/30.0)
        fps = int(round(1.0/DT))

        # -------------------- Calibrate on first frame ------------------------
        bl_est = BONE_LENGTHS.copy()
        theta_guess = get_default_joint_angles()

        # Good initial root: use pelvis marker from frame 0
        root_guess = markers_seq[0, 0, :]  # Pelvis is first in our loader order

        print("Calibrating bone lengths on first frame (AMASS)...")
        theta_cal, bl_cal, root_cal, *_ = lm_fit_markers_to_bones(
            bl_est, theta_guess, markers_seq[0], marker_bones=marker_bones,
            opt_joint_indices_list=[active_idx],
            use_segment=True,
            optimize_bones=True, optimize_root=True, root_init=root_guess,
            max_iters=30, tolerance=1e-3,
            angle_delta=angle_delta, length_delta=length_delta, root_delta=root_delta,
            lm_lambda0=5e-3, lm_lambda_factor=2.0,
            angle_step_clip=angle_step_clip, length_step_clip=length_step_clip, root_step_clip=root_step_clip,
            angle_reg=1.0, bone_reg=5.0, root_reg=0.5,
            marker_weights=np.ones(markers_seq.shape[1], dtype=float),
            joint_limits=(lower_lim, upper_lim),
            verbose=False,
            # Fixed correspondences (no auto-assign), so the flags below are ignored.
            auto_assign_bones=False, assign_topk=ASSIGN_TOPK,
            geom=SOLVER_GEOM, bone_radii=None,
            marker_batch_size=BATCH_SZ, reassign_every=REASSIGN_EVERY,
            fast_vectorized=FAST_VEC, rng_seed=0
        )

        bl_est     = bl_cal.copy()
        theta_prev = theta_cal.copy()
        root_prev  = root_cal.copy()

        # Storage for playback / analysis
        fit_thetas, fit_roots, per_frame_rmse = [], [], []

        print("Retargeting AMASS sequence...")
        t0 = time.time()
        for fidx in range(N_FRAMES):
            markers_t = markers_seq[fidx]

            theta_fit, bl_fit, root_fit, *_ = lm_fit_markers_to_bones(
                bl_est, theta_prev, markers_t, marker_bones=marker_bones,
                opt_joint_indices_list=[active_idx],
                use_segment=True,
                optimize_bones=False, optimize_root=True, root_init=root_prev,
                max_iters=18, tolerance=8e-4,
                angle_delta=angle_delta, length_delta=length_delta, root_delta=root_delta,
                lm_lambda0=5e-3, lm_lambda_factor=2.0,
                angle_step_clip=angle_step_clip, length_step_clip=length_step_clip, root_step_clip=root_step_clip,
                angle_reg=1.0, bone_reg=5.0, root_reg=0.5,
                marker_weights=np.ones(markers_t.shape[0], dtype=float),
                joint_limits=(lower_lim, upper_lim),
                verbose=False,
                auto_assign_bones=False, assign_topk=ASSIGN_TOPK,  # fixed correspondences
                geom=SOLVER_GEOM, bone_radii=None,
                marker_batch_size=BATCH_SZ, reassign_every=REASSIGN_EVERY,
                fast_vectorized=FAST_VEC, rng_seed=1+fidx
            )

            theta_prev, root_prev = theta_fit, root_fit
            fit_thetas.append(theta_fit); fit_roots.append(root_fit)

            # RMSE on segments for the known correspondences
            jp_fit, _ = get_joint_positions_and_orientations(bl_est, theta_fit, root_pos=root_fit)
            rs = build_residual_stack_hard_geom(jp_fit, markers_t, marker_bones, geom="segment", bone_radii=None)
            rs = rs.reshape(-1, 3)
            rmse = float(np.sqrt(np.mean(np.sum(rs*rs, axis=1))))
            per_frame_rmse.append(rmse)

        elapsed = time.time() - t0
        print(f"AMASS retargeting: {N_FRAMES} frames in {elapsed:.3f}s  ({1000.0*elapsed/N_FRAMES:.1f} ms/frame)")
        print(f"RMSE mean={np.mean(per_frame_rmse):.4f} m, median={np.median(per_frame_rmse):.4f} m")

        # -------------------- Animation (fitted skeleton + observed markers) ---
        fig = plt.figure(figsize=(8, 9))
        ax  = fig.add_subplot(111, projection='3d')

        def animate_frame(i):
            ax.clear()
            jp_fit, _ = get_joint_positions_and_orientations(bl_est, fit_thetas[i], root_pos=fit_roots[i])
            # wireframe skeleton
            draw_skeleton_wire(ax, jp_fit, color='black', lw=2.5, alpha=1.0)
            ax.scatter(jp_fit[:, 0], jp_fit[:, 1], jp_fit[:, 2], color='k', s=25)

            # observed markers (AMASS)
            m = markers_seq[i]
            ax.scatter(m[:, 0], m[:, 1], m[:, 2], marker='x', s=30, color='C1', alpha=0.9)

            ax.set_title(f"AMASS retarget: frame {i+1}/{N_FRAMES} | t={times[i]:.2f}s | RMSE={per_frame_rmse[i]:.3f} m")
            ax.set_xlabel('X (forward)'); ax.set_ylabel('Y (left)'); ax.set_zlabel('Z (up)')
            ax.set_box_aspect([1, 1, 1]); set_axes_equal(ax)

        SAVE_VIDEO = True
        ani = FuncAnimation(fig, animate_frame, frames=N_FRAMES, interval=1000/fps, repeat=False)
        if SAVE_VIDEO:
            save_animation(ani, "amass_retarget", fps=fps, dpi=150)
        else:
            plt.show()
        plt.close(fig)

    else:
        MOTION = "walk"   # "walk" | "run" | "march" | "sidestep" | "turn" | "squat" | "jump" | "stairs"
        angles_fn, root_fn = make_motion(MOTION, speed=SPEED, f=GAIT_F)  # add extra kwargs to tweak

        # skeleton + radii
        bone_radii = default_bone_radii()
        bl_gt = BONE_LENGTHS.copy()

        # marker template (consistent across frames)
        markers_per_bone = 10
        template = make_marker_template(BONES_IDX, markers_per_bone=markers_per_bone, geom=GT_GEOM, seed=123)


        semantic_priors = {}

        # storage
        gt_thetas = []
        gt_roots  = []
        gt_positions = []
        fit_thetas = []
        fit_roots  = []
        per_frame_rmse = []

        # ---------- initialize with frame 0 ----------
        t0 = 0.0
        # theta_gt0 = gait_angles(t0, f=GAIT_F)
        # root_gt0  = gait_root(t0, speed=SPEED, f=GAIT_F)
        # frame 0 init:
        theta_gt0 = angles_fn(t0)
        root_gt0  = root_fn(t0)


        jp_gt0, _ = get_joint_positions_and_orientations(bl_gt, theta_gt0, root_pos=root_gt0)
        # jitter_tangent_std controls the spiral position noise
        jitter_tangent_std = 0 if GT_GEOM == "segment" else 5.0
        markers0_clean, _ = render_markers_from_template(jp_gt0, template, bone_radii=bone_radii, jitter_tangent_std=jitter_tangent_std, seed=42)
        rng = np.random.default_rng(0)
        markers0 = markers0_clean + rng.normal(0.0, MEAS_NOISE_STD, markers0_clean.shape)

        # Stage A: one-time bone-length calibration on first frame (optional)
        theta_guess = get_default_joint_angles()
        root_guess  = np.array([0.0, 0.0, 0.0])

        print("Calibrating bone lengths on first frame...")
        theta_cal, bl_cal, root_cal, *_ = lm_fit_markers_to_bones(
            BONE_LENGTHS, theta_guess, markers0, marker_bones=None,
            opt_joint_indices_list=[active_idx],
            use_segment=True,
            optimize_bones=True, optimize_root=True, root_init=root_guess,
            max_iters=25, tolerance=1e-3,
            angle_delta=8e-4, length_delta=8e-4, root_delta=1e-3,
            lm_lambda0=5e-3, lm_lambda_factor=2.0,
            angle_step_clip=np.deg2rad(10.0),
            length_step_clip=0.015, root_step_clip=0.05,
            angle_reg=1.0, bone_reg=5.0, root_reg=0.5,
            marker_weights=np.ones(len(markers0)),
            joint_limits=(lower_lim, upper_lim),
            verbose=False,
            auto_assign_bones=True, assign_topk=1,
            assign_soft_sigma_factor=0.10,
            assign_enable_gate=True, assign_distance_gate_abs=None, assign_distance_gate_factor=0.8,
            assign_enable_hysteresis=True, assign_hysteresis_margin=0.08,
            assign_enable_temporal_smoothing=False,
            assign_semantic_priors=semantic_priors,
            strategy="lm+linesearch", line_search_scales=(1.0, 0.5, 0.25, 0.1),
            allow_trial_reassign=False,
            geom=GT_GEOM, bone_radii=bone_radii if DRAW_SOLIDS else None,
            marker_batch_size=BATCH_SZ, reassign_every=REASSIGN_EVERY, fast_vectorized=FAST_VEC, rng_seed=0
        )
        # Use calibrated bone lengths for all frames
        bl_est = bl_cal.copy()
        theta_prev = theta_cal.copy()
        root_prev  = root_cal.copy()

        # ---------- per-frame loop ----------
        print("Tracking walking sequence...")
        start_time = time.time()
        for fidx in range(N_FRAMES):
            t = fidx * DT

            # GT
            # theta_gt = gait_angles(t, f=GAIT_F)
            # root_gt  = gait_root(t, speed=SPEED, f=GAIT_F)
            theta_gt = angles_fn(t)
            root_gt  = root_fn(t)
            jp_gt, _ = get_joint_positions_and_orientations(bl_gt, theta_gt, root_pos=root_gt)
            markers_t_clean, _ = render_markers_from_template(jp_gt, template, bone_radii=bone_radii, jitter_tangent_std=jitter_tangent_std, seed=fidx)  # deterministic jitter across frames
            rng = np.random.default_rng(fidx)
            markers_t = markers_t_clean + rng.normal(0.0, MEAS_NOISE_STD, markers_t_clean.shape)

            # IK on this frame (warm-start from previous)
            theta_fit, bl_fit, root_fit, *_ = lm_fit_markers_to_bones(
                bl_est, theta_prev, markers_t, marker_bones=None,
                opt_joint_indices_list=[active_idx],
                use_segment=True,
                optimize_bones=False, optimize_root=True, root_init=root_prev,
                max_iters=18, tolerance=8e-4,
                angle_delta=8e-4, length_delta=8e-4, root_delta=1e-3,
                lm_lambda0=5e-3, lm_lambda_factor=2.0,
                angle_step_clip=np.deg2rad(10.0),
                length_step_clip=0.015, root_step_clip=0.05,
                angle_reg=1.0, bone_reg=5.0, root_reg=0.5,
                marker_weights=np.ones(len(markers_t)),
                joint_limits=(lower_lim, upper_lim),
                verbose=False,
                auto_assign_bones=True, assign_topk=1,
                assign_soft_sigma_factor=0.10,
                assign_enable_gate=True, assign_distance_gate_abs=None, assign_distance_gate_factor=1.0,
                assign_enable_hysteresis=True, assign_hysteresis_margin=0.08,
                assign_enable_temporal_smoothing=False,
                assign_semantic_priors=semantic_priors,
                strategy="lm+linesearch", line_search_scales=(1.0, 0.5, 0.25, 0.1),
                allow_trial_reassign=False,
                geom=GT_GEOM, bone_radii=bone_radii if DRAW_SOLIDS else None,
                marker_batch_size=BATCH_SZ, reassign_every=REASSIGN_EVERY, fast_vectorized=FAST_VEC, rng_seed=1+fidx
            )

            # save / warm-start
            theta_prev = theta_fit
            root_prev  = root_fit

            gt_thetas.append(theta_gt); gt_roots.append(root_gt); gt_positions.append(jp_gt)
            fit_thetas.append(theta_fit); fit_roots.append(root_fit)

            # compute per-frame RMSE (project to surfaces if thick geometry)
            jp_fit, _ = get_joint_positions_and_orientations(bl_est, theta_fit, root_pos=root_fit)
            corr = robust_assign_markers(
                markers_t, jp_fit, BONES_IDX, prev_state=None,
                bone_lengths=bl_est, topk=1, soft_sigma_factor=0.1,
                distance_gate_abs=None, distance_gate_factor=1.0, enable_gate=True,
                hysteresis_margin=0.10, enable_hysteresis=True,
                temporal_smoothing=0.0, enable_temporal_smoothing=False,
                semantic_priors=semantic_priors,
                geom=GT_GEOM, bone_radii=bone_radii if DRAW_SOLIDS else None
            )
            if corr['mode'] == 'hard':
                if DRAW_SOLIDS:
                    # compute distances to surface
                    dists = []
                    for k, (ja, jb) in enumerate(corr['hard']):
                        bi = BONE_PAIR_TO_INDEX[(ja, jb)]
                        R = float(bone_radii[bi]) if DRAW_SOLIDS else 0.0
                        if GT_GEOM == "capsule":
                            s = closest_point_on_capsule_surface(markers_t[k], jp_fit[ja], jp_fit[jb], R)
                        elif GT_GEOM == "cylinder":
                            s = closest_point_on_capped_cylinder_surface(markers_t[k], jp_fit[ja], jp_fit[jb], R)
                        else:
                            # segment distance
                            v = _closest_point_on_segment_pointwise(markers_t[k], jp_fit[ja], jp_fit[jb])
                            s = v
                        dists.append(np.linalg.norm(markers_t[k] - s))
                    rmse = (np.mean(np.square(dists)))**0.5
                else:
                    # line/segment residual
                    rs = build_residual_stack_hard_geom(jp_fit, markers_t, corr['hard'], geom="segment", bone_radii=None).reshape(-1,3)
                    rmse = np.sqrt(np.mean(np.sum(rs*rs,axis=1)))
            else:
                # soft mode not used here
                rmse = np.nan
            per_frame_rmse.append(rmse)

        elapsed = time.time() - start_time
        print(f"Tracking time for {N_FRAMES} frames: {elapsed:.3f}s  (avg {1000.0*elapsed/N_FRAMES:.1f} ms/frame)")
        print(f"RMSE mean={np.mean(per_frame_rmse):.4f} m,  median={np.median(per_frame_rmse):.4f} m")

        # ------------- Animation of frames: GT vs Fitted ----------------
        fig = plt.figure(figsize=(8, 9))
        ax = fig.add_subplot(111, projection='3d')

        def animate_frame(i):
            ax.clear()
            # GT
            jp_gt = gt_positions[i]
            plot_skeleton(ax, jp_gt, [None]*len(jp_gt),
                          markers=None, marker_bones=None, show_axes=False,
                          title=f'Walking: frame {i+1}/{N_FRAMES}  |  RMSE={per_frame_rmse[i]:.3f} m',
                          draw_solids=DRAW_SOLIDS, bone_radii=bone_radii if DRAW_SOLIDS else None,
                          clear=True, joint_color='gray', wire_color='gray', wire_alpha=0.6)
            # Fitted
            jp_fit, jo_fit = get_joint_positions_and_orientations(bl_est, fit_thetas[i], root_pos=fit_roots[i])
            draw_skeleton_wire(ax, jp_fit, color='black', lw=2.5, alpha=1.0)
            ax.scatter(jp_fit[:,0], jp_fit[:,1], jp_fit[:,2], color='k', s=25)
            # Markers for this frame (from GT)
            markers_i_clean, _ = render_markers_from_template(jp_gt, template, bone_radii=bone_radii, jitter_tangent_std=jitter_tangent_std, seed=i)
            rng = np.random.default_rng(i)
            markers_t = markers_i_clean + rng.normal(0.0, MEAS_NOISE_STD, markers_i_clean.shape)
            ax.scatter(markers_t[:,0], markers_t[:,1], markers_t[:,2], marker='x', s=28, color='C1', alpha=0.9)
            ax.set_xlabel('X (forward)'); ax.set_ylabel('Y (left)'); ax.set_zlabel('Z (up)')
            ax.set_box_aspect([1,1,1]); set_axes_equal(ax)

        SAVE_VIDEO = True
        ani = FuncAnimation(fig, animate_frame, frames=N_FRAMES, interval=1000/fps, repeat=False)
        if SAVE_VIDEO:
            save_animation(ani, MOTION, fps=int(round(1/DT)), dpi=150)
        else:
            plt.show()

        plt.close(fig)  # optional tidy-up