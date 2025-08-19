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
# print(f"SPINE_TOP_YAW: ", SPINE_TOP_YAW)
# print(f"RIGHT_SHOULDER_YAW: ", RIGHT_SHOULDER_YAW)
BONES_IDX = [
    (PELVIS, SPINE_TOP), (SPINE_TOP, NECK_TOP), (NECK_TOP, HEAD_TOP),                      # upper body
    (SPINE_TOP, RIGHT_SHOULDER), (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_HAND), # right arm
    (SPINE_TOP, LEFT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_HAND),      # left arm
    (PELVIS, RIGHT_HIP), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_FOOT),                # right leg
    (PELVIS, LEFT_HIP), (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_FOOT)                      # left leg
]

BONE_LENGTHS = {
    'spine': 0.5, 'neck': 0.1, 'head': 0.1,
    'upper_arm': 0.3, 'lower_arm': 0.25,
    'upper_leg': 0.4, 'lower_leg': 0.4,
    'shoulder_offset': 0.2, 'hip_offset': 0.1
}

# Keys of bone lengths to optimize
BONE_LENGTH_KEYS_TO_OPTIMIZE = ['upper_arm', 'lower_arm', 'upper_leg', 'lower_leg', 'neck']

def get_default_bone_to_optimize_lengths_vec():
    return np.array([BONE_LENGTHS[key] for key in BONE_LENGTH_KEYS_TO_OPTIMIZE])

def update_bone_lengths_from_vec(bone_lengths, vec):
    for k, v in zip(BONE_LENGTH_KEYS_TO_OPTIMIZE, vec):
        bone_lengths[k] = v
    return bone_lengths

# Target joint: [all joints that will affect target joints]
DEFAULT_OPT_JOINTS = {
    HEAD_TOP:    [SPINE_TOP_YAW, SPINE_TOP_PITCH, SPINE_TOP_ROLL, NECK_TOP_YAW, NECK_TOP_PITCH, NECK_TOP_ROLL],
    RIGHT_HAND:  [SPINE_TOP_YAW, SPINE_TOP_PITCH, RIGHT_SHOULDER_YAW, RIGHT_SHOULDER_PITCH, RIGHT_SHOULDER_ROLL,
                  RIGHT_ELBOW_YAW, RIGHT_ELBOW_PITCH, RIGHT_ELBOW_ROLL],
    LEFT_HAND:   [SPINE_TOP_YAW, SPINE_TOP_PITCH, LEFT_SHOULDER_YAW, LEFT_SHOULDER_PITCH, LEFT_SHOULDER_ROLL,
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
  
def _deg(a):  # degrees -> radians
    return np.deg2rad(a)

def get_default_joint_limits():
    """
    Returns (lower, upper) arrays of shape (48,) for yaw, pitch, roll per joint.
    Limits are conservative human-like ranges; tweak as needed.
    """
    lower = -np.pi * np.ones(48, dtype=float)
    upper =  np.pi * np.ones(48, dtype=float)

    def set_limits(joint, yaw_range, pitch_range, roll_range):
        y0, y1 = _deg(yaw_range[0]), _deg(yaw_range[1])
        p0, p1 = _deg(pitch_range[0]), _deg(pitch_range[1])
        r0, r1 = _deg(roll_range[0]), _deg(roll_range[1])
        i = 3 * joint
        lower[i + 0], upper[i + 0] = y0, y1  # yaw (Z)
        lower[i + 1], upper[i + 1] = p0, p1  # pitch (Y)
        lower[i + 2], upper[i + 2] = r0, r1  # roll (X)

    # Root pelvis: generous
    set_limits(PELVIS,       (-180, 180), (-90, 90),  (-90, 90))
    # Spine / neck / head: moderate
    set_limits(SPINE_TOP,    (-60,  60),  (-45, 45),  (-45, 45))
    set_limits(NECK_TOP,     (-80,  80),  (-60, 60),  (-60, 60))
    set_limits(HEAD_TOP,     (-80,  80),  (-60, 60),  (-60, 60))
    # Shoulders (ball joints)
    set_limits(RIGHT_SHOULDER, (-150, 150), (-150, 150), (-100, 100))
    set_limits(LEFT_SHOULDER,  (-150, 150), (-150, 150), (-100, 100))
    # Elbows (mostly hinge on pitch; keep yaw/roll tighter)
    set_limits(RIGHT_ELBOW,  (-45, 45), (0, 150), (-45, 45))
    set_limits(LEFT_ELBOW,   (-45, 45), (0, 150), (-45, 45))
    # Hands/wrists (liberal)
    set_limits(RIGHT_HAND,   (-90, 90), (-90, 90), (-90, 90))
    set_limits(LEFT_HAND,    (-90, 90), (-90, 90), (-90, 90))
    # Hips (ball joints)
    set_limits(RIGHT_HIP,    (-70, 70), (-120, 120), (-50, 50))
    set_limits(LEFT_HIP,     (-70, 70), (-120, 120), (-50, 50))
    # Knees (mostly hinge on pitch)
    set_limits(RIGHT_KNEE,   (-30, 30), (0, 150), (-30, 30))
    set_limits(LEFT_KNEE,    (-30, 30), (0, 150), (-30, 30))
    # Feet/ankles
    set_limits(RIGHT_FOOT,   (-45, 45), (-45, 45), (-30, 30))
    set_limits(LEFT_FOOT,    (-45, 45), (-45, 45), (-30, 30))

    return lower, upper


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

def residual_point_to_line(marker, pa, pb):
    """
    Residual is the perpendicular component of (marker - pa) to the infinite line through pa->pb.
    Returns a 3D vector r s.t. ||r|| is the point-to-line distance.
    """
    v = pb - pa
    L = np.linalg.norm(v)
    if L < 1e-12:
        # degeneracy: treat like point-to-point to pa
        return marker - pa
    u = v / L
    P = np.eye(3) - np.outer(u, u)      # projector onto the plane orthogonal to u
    return P @ (marker - pa)

def residual_point_to_segment(marker, pa, pb):
    """
    Residual to the closest point on the finite segment [pa, pb].
    Returns a 3D vector whose norm is the point-to-segment distance.
    """
    v = pb - pa
    L2 = v @ v
    if L2 < 1e-12:
        return marker - pa
    t = (marker - pa) @ v / L2
    t = np.clip(t, 0.0, 1.0)
    closest = pa + t * v
    return marker - closest

def overlay_marker_projections(ax, joint_positions, markers, marker_bones, color='C1', alpha=0.6):
    for m, (ja, jb) in zip(markers, marker_bones):
        pa = joint_positions[ja]; pb = joint_positions[jb]
        v  = pb - pa
        L2 = np.dot(v, v)
        if L2 < 1e-12:
            continue
        t = np.clip(np.dot(m - pa, v) / L2, 0.0, 1.0)
        closest = pa + t * v
        ax.plot([m[0], closest[0]], [m[1], closest[1]], [m[2], closest[2]], alpha=alpha, linewidth=1.5, color=color)

from collections import defaultdict

# ---- Scale & groups ----------------------------------------------------------

def compute_skeleton_scale(bone_lengths):
    """A rough body scale (meters). Used for relative gates/sigmas."""
    return (
        bone_lengths['upper_leg'] + bone_lengths['lower_leg'] +
        bone_lengths['spine'] + bone_lengths['neck'] + bone_lengths['head']
    )

def bone_groups(bones_idx=BONES_IDX):
    """
    Return dict of group_name -> set of bone indices (by index into BONES_IDX).
    Groups: torso, head, arms, legs, left_arm, right_arm, left_leg, right_leg,
            upper_body, lower_body, all
    """
    g = defaultdict(set)
    for bi, (ja, jb) in enumerate(bones_idx):
        # torso/head
        if (ja, jb) in [(PELVIS, SPINE_TOP), (SPINE_TOP, NECK_TOP)]: g['torso'].add(bi)
        if (ja, jb) == (NECK_TOP, HEAD_TOP): g['head'].add(bi)
        # arms
        if ja == SPINE_TOP and jb in (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HAND): g['right_arm'].add(bi)
        if ja == SPINE_TOP and jb in (LEFT_SHOULDER, LEFT_ELBOW, LEFT_HAND): g['left_arm'].add(bi)
        if ja in (RIGHT_SHOULDER, RIGHT_ELBOW): g['right_arm'].add(bi)
        if ja in (LEFT_SHOULDER, LEFT_ELBOW): g['left_arm'].add(bi)
        # legs
        if ja == PELVIS and jb in (RIGHT_HIP, LEFT_HIP):  # hip offsets
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
    """
    Normalize semantic_priors to: {marker_index: set(bone_indices)}.
    Accepts region names (from bone_groups) or direct bone indices/tuples.
    """
    if not semantic_priors:
        return {}
    groups = bone_groups(bones_idx)
    out = {}
    for mi, allowed in semantic_priors.items():
        s = set()
        for item in (allowed if isinstance(allowed, (list, tuple, set)) else [allowed]):
            if isinstance(item, str) and item in groups:
                s |= groups[item]
            elif isinstance(item, int):
                s.add(item)
            elif isinstance(item, tuple) and len(item) == 2:
                # (ja, jb) tuple -> find its index
                try:
                    s.add(bones_idx.index(item))
                except ValueError:
                    pass
        out[int(mi)] = s
    return out

# ---- Distances & projections -------------------------------------------------

def _closest_point_on_segment(marker, pa, pb):
    v = pb - pa
    L2 = v @ v
    if L2 < 1e-12:
        return pa
    t = (marker - pa) @ v / L2
    t = np.clip(t, 0.0, 1.0)
    return pa + t * v

def _point_to_segment_distance(marker, pa, pb):
    return np.linalg.norm(marker - _closest_point_on_segment(marker, pa, pb))

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
    prev_state=None,                 # dict with keys: 'hard', 'weights' (per marker)
    bone_lengths=BONE_LENGTHS,
    # toggles & params:
    topk=1,                          # K=1 -> hard; K>1 -> soft
    soft_sigma_factor=0.1,           # sigma = factor * body_scale
    distance_gate_abs=None,          # e.g., 0.2 (meters). None to disable.
    distance_gate_factor=1.0,        # gate = factor * body_scale
    enable_gate=True,                # toggle gating
    hysteresis_margin=0.10,          # keep prev bone if d_prev <= (1+margin)*d_best
    enable_hysteresis=True,
    temporal_smoothing=0.0,          # [0..1], blend weights with previous
    enable_temporal_smoothing=False,
    semantic_priors=None             # dict: marker_idx -> allowed bones/group names
):
    """
    Returns:
      corr = {
        'mode': 'hard' or 'soft',
        'hard': list of (ja, jb) for K=1 case,
        'cands': list of lists of candidate bone indices per marker,
        'weights': list of arrays (len=Ki) per marker (sum to 1),
        'state': {'hard': [...], 'weights': [dict(bi->w), ...]}
      }
    """
    n_bones = len(bones_idx)
    scale = compute_skeleton_scale(bone_lengths)
    sigma = max(1e-6, soft_sigma_factor * scale)
    gate = (distance_gate_abs if (enable_gate and distance_gate_abs is not None)
            else (distance_gate_factor * scale if enable_gate else np.inf))
    allowed_by_marker = expand_semantic_priors(semantic_priors, bones_idx)

    # previous state unpack
    prev_hard = prev_state.get('hard') if prev_state else None
    prev_wts  = prev_state.get('weights') if prev_state else None  # list of dicts {bone_idx: weight}

    cands = []
    weights = []
    hard_pairs = []

    for mi, m in enumerate(np.asarray(markers)):
        # Candidate set with optional semantic filtering
        allowed = allowed_by_marker.get(mi, None)  # None -> all
        dists = []
        idxs  = []
        for bi, (ja, jb) in enumerate(bones_idx):
            if (allowed is not None) and (bi not in allowed):
                continue
            pa, pb = joint_positions[ja], joint_positions[jb]
            d = (_point_to_segment_distance(m, pa, pb) if use_segment
                 else _point_to_line_distance(m, pa, pb))
            if not enable_gate or d <= gate:
                dists.append(d); idxs.append(bi)

        # If nothing passes gate, fall back to global nearest bone
        if not idxs:
            best = None; best_d = np.inf; best_bi = None
            for bi, (ja, jb) in enumerate(bones_idx):
                pa, pb = joint_positions[ja], joint_positions[jb]
                d = (_point_to_segment_distance(m, pa, pb) if use_segment
                     else _point_to_line_distance(m, pa, pb))
                if d < best_d: best_d, best_bi = d, bi
            idxs = [best_bi]; dists = [best_d]

        # Sort by distance, take topK
        order = np.argsort(dists)
        idxs = [idxs[i] for i in order][:max(1, topk)]
        dists = [dists[i] for i in order][:max(1, topk)]

        # Hysteresis: keep previous hard bone if close enough
        if enable_hysteresis and prev_hard is not None and mi < len(prev_hard) and prev_hard[mi] is not None:
            prev_bi = prev_hard[mi]
            # ensure prev_bi in candidate set
            if prev_bi not in idxs:
                # compute its distance (only if in semantic/gate? we allow it)
                ja, jb = bones_idx[prev_bi]
                pa, pb = joint_positions[ja], joint_positions[jb]
                d_prev = (_point_to_segment_distance(m, pa, pb) if use_segment
                          else _point_to_line_distance(m, pa, pb))
                # compare to best
                d_best = dists[0]
                if d_prev <= (1.0 + hysteresis_margin) * d_best:
                    # keep previous: override to hard stick
                    idxs = [prev_bi]; dists = [d_prev]

        # Soft weights (Gaussian on distance); K=1 gives weight=1.
        ws = np.exp(-0.5 * (np.array(dists) / sigma) ** 2)
        if ws.sum() <= 1e-12:
            ws = np.ones_like(ws)
        ws = ws / ws.sum()

        # Temporal smoothing on weight dictionaries
        if enable_temporal_smoothing and prev_wts is not None and mi < len(prev_wts) and prev_wts[mi]:
            # bring previous weights into current candidate set
            prev_dict = prev_wts[mi]
            merged = defaultdict(float)
            # add current
            for bi, w in zip(idxs, ws):
                merged[bi] += (1.0 - temporal_smoothing) * w
            # add previous (only for same bones; or include even if not in current)
            for bi, w in prev_dict.items():
                merged[bi] += temporal_smoothing * w
            # renormalize and project back to current candidate list (extend if necessary)
            items = sorted(merged.items(), key=lambda x: -x[1])
            # optionally trim to topk to keep model size bounded
            items = items[:max(1, topk)]
            idxs = [bi for bi, _ in items]
            ws   = np.array([w for _, w in items], dtype=float)
            ws = ws / ws.sum()

        # pick hard pair for visualization/fallback
        hard_bi = idxs[int(np.argmax(ws))]
        hard_pairs.append(bones_idx[hard_bi])

        cands.append(idxs)
        weights.append(ws)

    # Build next prev_state: store hard and dict weights
    next_state = {'hard': [], 'weights': []}
    for mi, (idxs, ws) in enumerate(zip(cands, weights)):
        # next_state['hard'].append(bones_idx[idxs[int(np.argmax(ws))]])
        best_idx = idxs[int(np.argmax(ws))]
        next_state['hard'].append(best_idx)  # store integer index
        next_state['weights'].append({bi: float(w) for bi, w in zip(idxs, ws)})

    return {
        'mode': 'soft' if topk > 1 else 'hard',
        'hard': hard_pairs,
        'cands': cands,
        'weights': weights,
        'state': next_state
    }

# ---- Residual builder for hard/soft -----------------------------------------

def build_residual_stack_hard(jp, markers, marker_bones, use_segment=True):
    K = len(markers)
    res = np.zeros((3 * K,), dtype=float)
    for k, (ja, jb) in enumerate(marker_bones):
        pa = jp[ja]; pb = jp[jb]
        if use_segment:
            r = residual_point_to_segment(markers[k], pa, pb)
        else:
            r = residual_point_to_line(markers[k], pa, pb)
        res[3*k:3*k+3] = r
    return res

def build_residual_stack_soft(jp, markers, bones_idx, cands, weights, use_segment=True):
    """
    Soft residual: r_k = m_k - sum_i w_i * closest_i
    """
    K = len(markers)
    res = np.zeros((3 * K,), dtype=float)
    for k in range(K):
        m = markers[k]
        idxs = cands[k]; ws = weights[k]
        assert len(idxs) == len(ws) and len(ws) >= 1
        closest_sum = np.zeros(3)
        for bi, w in zip(idxs, ws):
            ja, jb = bones_idx[bi]
            pa, pb = jp[ja], jp[jb]
            cp = _closest_point_on_segment(m, pa, pb) if use_segment else (pa + pb) / 2.0  # line case: could orth-project
            closest_sum += w * cp
        res[3*k:3*k+3] = m - closest_sum
    return res

def overlay_marker_soft_projections(ax, jp, markers, bones_idx, cands, weights, use_segment=True, color='C2', alpha=0.6):
    for k, m in enumerate(markers):
        # weighted closest point
        closest_sum = np.zeros(3)
        for bi, w in zip(cands[k], weights[k]):
            ja, jb = bones_idx[bi]
            pa, pb = jp[ja], jp[jb]
            cp = _closest_point_on_segment(m, pa, pb) if use_segment else (pa + pb) / 2.0
            closest_sum += w * cp
        ax.plot([m[0], closest_sum[0]], [m[1], closest_sum[1]], [m[2], closest_sum[2]], alpha=alpha, linewidth=1.5, color=color)

def lm_fit_markers_to_bones(
    bone_lengths,
    joint_angles,
    markers,
    marker_bones=None,            # optional when using auto/robust assignment
    opt_joint_indices_list=None,
    use_segment=True,
    optimize_bones=False,
    max_iters=100,
    tolerance=1e-3,
    angle_delta=1e-3,
    length_delta=1e-3,
    lm_lambda0=1e-2,
    lm_lambda_factor=2.0,
    lm_lambda_min=1e-6,
    lm_lambda_max=1e+2,
    angle_step_clip=np.deg2rad(12.0),
    bone_clip=(0.05, 2.0),
    angle_reg=1.0,
    bone_reg=5.0,
    marker_weights=None,
    joint_limits=None,
    verbose=False,
    # --- NEW robust assignment controls ---
    auto_assign_bones=False,
    assign_topk=1,                      # >1 enables soft mode
    assign_soft_sigma_factor=0.1,
    assign_distance_gate_abs=None,
    assign_distance_gate_factor=1.0,
    assign_enable_gate=True,
    assign_enable_hysteresis=True,
    assign_hysteresis_margin=0.10,
    assign_enable_temporal_smoothing=False,
    assign_temporal_smoothing=0.0,
    assign_semantic_priors=None         # dict: marker_idx -> list of group names or bone indices
):
    """
    Returns: (theta, bone_lengths_final, angles_history, bone_length_history)
    """
    theta = np.array(joint_angles, dtype=float).copy()

    # bone-length block ...
    if optimize_bones:
        bl_keys = BONE_LENGTH_KEYS_TO_OPTIMIZE
    else:
        bl_keys = []
    bl_vec = np.array([bone_lengths[k] for k in bl_keys], dtype=float)

    n_joints = theta.size
    if opt_joint_indices_list is None:
        opt_joint_indices_list = [list(range(n_joints))]
    active_angle_idx = sorted(set(i for idxs in opt_joint_indices_list for i in idxs)) or list(range(n_joints))
    n_active = len(active_angle_idx)

    K = len(markers)
    markers = np.asarray(markers, dtype=float).reshape(K, 3)

    # weights
    if marker_weights is None:
        marker_weights = np.ones(K, dtype=float)
    w = np.asarray(marker_weights, dtype=float).clip(min=0.0)
    w_sqrt = np.sqrt(w)

    # limits
    if joint_limits is None:
        lower_lim = -np.inf * np.ones(n_joints)
        upper_lim =  np.inf * np.ones(n_joints)
    else:
        lower_lim, upper_lim = joint_limits

    angles_history = [theta.copy()]
    bone_length_history = [bl_vec.copy()]
    lm_lambda = float(lm_lambda0)

    def fk_positions(curr_theta, curr_bl_vec):
        bl_all = bone_lengths.copy()
        for k, v in zip(bl_keys, curr_bl_vec):
            bl_all[k] = v
        jp, _ = get_joint_positions_and_orientations(bl_all, curr_theta)
        return jp, bl_all

    # persistent assignment state across iters (for hysteresis/smoothing)
    assign_state = {'hard': None, 'weights': None}
    jp, bl_all = fk_positions(theta, bl_vec)

    # initial correspondences
    if auto_assign_bones or (marker_bones is None):
        corr = robust_assign_markers(
            markers, jp, BONES_IDX, use_segment=use_segment, prev_state=assign_state,
            bone_lengths=bl_all,
            topk=assign_topk,
            soft_sigma_factor=assign_soft_sigma_factor,
            distance_gate_abs=assign_distance_gate_abs,
            distance_gate_factor=assign_distance_gate_factor,
            enable_gate=assign_enable_gate,
            hysteresis_margin=assign_hysteresis_margin,
            enable_hysteresis=assign_enable_hysteresis,
            temporal_smoothing=assign_temporal_smoothing,
            enable_temporal_smoothing=assign_enable_temporal_smoothing,
            semantic_priors=assign_semantic_priors
        )
        assign_state = corr['state']
        if corr['mode'] == 'hard':
            e = build_residual_stack_hard(jp, markers, corr['hard'], use_segment=use_segment)
        else:
            e = build_residual_stack_soft(jp, markers, BONES_IDX, corr['cands'], corr['weights'], use_segment=use_segment)
    else:
        corr = {'mode': 'hard', 'hard': list(marker_bones)}
        e = build_residual_stack_hard(jp, markers, corr['hard'], use_segment=use_segment)

    # initial error
    prev_err = np.linalg.norm(np.repeat(w_sqrt, 3) * e)

    for it in range(max_iters):
        n_bones = bl_vec.size
        J_theta = np.zeros((3 * K, n_active))
        J_bl    = np.zeros((3 * K, n_bones))

        # E-step: refresh correspondences at current (theta, bl)
        jp_base, bl_all = fk_positions(theta, bl_vec)
        if auto_assign_bones or (marker_bones is None):
            corr = robust_assign_markers(
                markers, jp_base, BONES_IDX, use_segment=use_segment, prev_state=assign_state,
                bone_lengths=bl_all,
                topk=assign_topk,
                soft_sigma_factor=assign_soft_sigma_factor,
                distance_gate_abs=assign_distance_gate_abs,
                distance_gate_factor=assign_distance_gate_factor,
                enable_gate=assign_enable_gate,
                hysteresis_margin=assign_hysteresis_margin,
                enable_hysteresis=assign_enable_hysteresis,
                temporal_smoothing=assign_temporal_smoothing,
                enable_temporal_smoothing=assign_enable_temporal_smoothing,
                semantic_priors=assign_semantic_priors
            )
            assign_state = corr['state']

        # build residuals with correspondences FROZEN within this linearization
        if corr.get('mode', 'hard') == 'hard':
            e = build_residual_stack_hard(jp_base, markers, corr['hard'], use_segment=use_segment)
        else:
            e = build_residual_stack_soft(jp_base, markers, BONES_IDX, corr['cands'], corr['weights'], use_segment=use_segment)

        # ---- Angle Jacobian ----
        for c, j_idx in enumerate(active_angle_idx):
            orig = theta[j_idx]
            theta[j_idx] = orig + angle_delta
            jp_pert, _ = fk_positions(theta, bl_vec)
            if corr.get('mode', 'hard') == 'hard':
                e_pert = build_residual_stack_hard(jp_pert, markers, corr['hard'], use_segment=use_segment)
            else:
                e_pert = build_residual_stack_soft(jp_pert, markers, BONES_IDX, corr['cands'], corr['weights'], use_segment=use_segment)
            J_theta[:, c] = (e_pert - e) / angle_delta
            theta[j_idx] = orig

        # ---- Bone-length Jacobian ----
        for c in range(n_bones):
            orig = bl_vec[c]
            bl_vec[c] = orig + length_delta
            jp_pert, _ = fk_positions(theta, bl_vec)
            if corr.get('mode', 'hard') == 'hard':
                e_pert = build_residual_stack_hard(jp_pert, markers, corr['hard'], use_segment=use_segment)
            else:
                e_pert = build_residual_stack_soft(jp_pert, markers, BONES_IDX, corr['cands'], corr['weights'], use_segment=use_segment)
            J_bl[:, c] = (e_pert - e) / length_delta
            bl_vec[c] = orig

        # Stack Jacobian + weights
        J = np.hstack([J_theta, J_bl])
        e_weighted = e.copy()
        if not np.allclose(w, 1.0):
            for k in range(K):
                row = 3 * k
                J[row:row+3, :] *= w_sqrt[k]
                e_weighted[row:row+3] *= w_sqrt[k]

        JTJ = J.T @ J
        JTe = - (J.T @ e_weighted)
        D = np.diag(np.concatenate([angle_reg * np.ones(n_active), bone_reg * np.ones(n_bones)]))

        improved = False
        for _trial in range(8):
            A = JTJ + (lm_lambda ** 2) * D
            try:
                delta = np.linalg.solve(A, JTe)
            except np.linalg.LinAlgError:
                A = A + 1e-9 * np.eye(A.shape[0])
                delta = np.linalg.solve(A, JTe)

            d_theta = delta[:n_active]
            d_bl    = delta[n_active:]

            if angle_step_clip is not None and d_theta.size:
                max_step = np.max(np.abs(d_theta))
                if max_step > angle_step_clip:
                    scale = angle_step_clip / (max_step + 1e-12)
                    d_theta *= scale
                    d_bl    *= scale

            theta_new = theta.copy()
            theta_new[active_angle_idx] += d_theta
            theta_new = np.minimum(np.maximum(theta_new, lower_lim), upper_lim)

            if n_bones:
                bl_vec_new = np.clip(bl_vec + d_bl, bone_clip[0], bone_clip[1])
            else:
                bl_vec_new = bl_vec

            jp_new, bl_all_new = fk_positions(theta_new, bl_vec_new)
            # evaluate with same frozen correspondences
            if corr.get('mode', 'hard') == 'hard':
                e_new = build_residual_stack_hard(jp_new, markers, corr['hard'], use_segment=use_segment)
            else:
                e_new = build_residual_stack_soft(jp_new, markers, BONES_IDX, corr['cands'], corr['weights'], use_segment=use_segment)
            err_new = np.linalg.norm(np.repeat(w_sqrt, 3) * e_new)

            if verbose:
                print(f"iter {it:03d} trial: err_new={err_new:.6f}, prev_err={prev_err:.6f}, λ={lm_lambda:.2e} (mode={corr.get('mode','hard')}, topk={assign_topk})")

            if err_new < prev_err:
                theta = theta_new
                bl_vec = bl_vec_new
                prev_err = err_new
                lm_lambda = max(lm_lambda / lm_lambda_factor, lm_lambda_min)
                improved = True
                break
            else:
                lm_lambda = min(lm_lambda * lm_lambda_factor, lm_lambda_max)

        angles_history.append(theta.copy())
        bone_length_history.append(bl_vec.copy())

        if verbose:
            print(f"[LM markers] iter {it+1:03d}  weighted_err={prev_err:.6f}")

        if prev_err < tolerance:
            if verbose:
                print(f"[LM markers] converged in {it+1} iters, weighted_err={prev_err:.6f}")
            break

    # Final bone lengths
    bone_lengths_final = bone_lengths.copy()
    for k, v in zip(bl_keys, bl_vec):
        bone_lengths_final[k] = v

    return theta, bone_lengths_final, angles_history, bone_length_history


def make_active_dof_indices_human_like_hinges():
    """Allow torso/neck, shoulders, hips (full 3-DOF) + elbows/knees pitch-only."""
    active = []
    # Torso / neck
    active += [ SPINE_TOP_PITCH, SPINE_TOP_ROLL,
                 NECK_TOP_PITCH,  NECK_TOP_ROLL]
    # Shoulders (ball)
    active += [RIGHT_SHOULDER_YAW, RIGHT_SHOULDER_ROLL,
               LEFT_SHOULDER_YAW,  LEFT_SHOULDER_ROLL]
    # Hips (ball)
    active += [RIGHT_HIP_YAW, RIGHT_HIP_PITCH, RIGHT_HIP_ROLL,
               LEFT_HIP_YAW,  LEFT_HIP_PITCH,  LEFT_HIP_ROLL]
    # Elbows / knees: pitch-only (pure hinges)
    active += [RIGHT_ELBOW_PITCH, LEFT_ELBOW_PITCH, RIGHT_KNEE_PITCH, LEFT_KNEE_PITCH]
    # Wrists/ankles are frozen by default; add if needed
    return sorted(set(active))

def enforce_pure_hinges_in_limits(lower, upper, tight_deg=0.5):
    """
    Clamp elbow/knee yaw & roll to ~0 so they're effectively hinges around pitch.
    """
    eps = np.deg2rad(tight_deg)
    for j in (RIGHT_ELBOW, LEFT_ELBOW, RIGHT_KNEE, LEFT_KNEE):
        i = 3 * j
        # yaw (Z) and roll (X) -> tiny range around 0
        lower[i + 0], upper[i + 0] = -eps, eps
        lower[i + 2], upper[i + 2] = -eps, eps
    return lower, upper

def make_gt_angles():
    """Define a ground-truth posture (48-dim). Adjust as you like."""
    theta = get_default_joint_angles()

    # Torso & head
    theta[SPINE_TOP_YAW]   = np.deg2rad(10)
    theta[SPINE_TOP_PITCH] = np.deg2rad(5)
    theta[NECK_TOP_YAW]    = np.deg2rad(-10)
    theta[NECK_TOP_PITCH]  = np.deg2rad(5)

    # Arms: both forward, elbows flexed
    theta[RIGHT_SHOULDER_YAW] = np.deg2rad(35)
    # theta[RIGHT_ELBOW_PITCH]    = np.deg2rad(50)
    theta[LEFT_SHOULDER_YAW]  = np.deg2rad(-35)
    # theta[LEFT_ELBOW_PITCH]     = np.deg2rad(50)

    # Legs: slight bend
    theta[RIGHT_HIP_PITCH]  = np.deg2rad(-25)
    theta[RIGHT_KNEE_PITCH] = np.deg2rad(40)
    theta[LEFT_HIP_PITCH]   = np.deg2rad(-25)
    theta[LEFT_KNEE_PITCH]  = np.deg2rad(45)

    return theta

def _perp_basis(u):
    """Return two unit vectors perpendicular to u (3,)."""
    u = u / (np.linalg.norm(u) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n1 = np.cross(u, tmp)
    n1 /= (np.linalg.norm(n1) + 1e-12)
    n2 = np.cross(u, n1)
    n2 /= (np.linalg.norm(n2) + 1e-12)
    return n1, n2

def _point_to_segment_distance(marker, pa, pb):
    v = pb - pa
    L2 = v @ v
    if L2 < 1e-12:
        return np.linalg.norm(marker - pa)
    t = (marker - pa) @ v / L2
    t = np.clip(t, 0.0, 1.0)
    closest = pa + t * v
    return np.linalg.norm(marker - closest)

def _point_to_line_distance(marker, pa, pb):
    v = pb - pa
    L = np.linalg.norm(v)
    if L < 1e-12:
        return np.linalg.norm(marker - pa)
    u = v / L
    P = np.eye(3) - np.outer(u, u)
    return np.linalg.norm(P @ (marker - pa))

def assign_markers_to_bones(markers, joint_positions, bones_idx=BONES_IDX, use_segment=True):
    """
    Hard-assign each marker to the nearest bone under the current pose.
    Returns: (marker_bones, dists) where marker_bones[k] = (ja, jb)
    """
    markers = np.asarray(markers, dtype=float)
    assigned = []
    dists = []
    for m in markers:
        best = None
        best_d = np.inf
        for (ja, jb) in bones_idx:
            pa, pb = joint_positions[ja], joint_positions[jb]
            d = (_point_to_segment_distance(m, pa, pb) if use_segment
                 else _point_to_line_distance(m, pa, pb))
            if d < best_d:
                best_d = d
                best = (ja, jb)
        assigned.append(best)
        dists.append(best_d)
    return assigned, np.array(dists)

def sample_markers_on_bones(joint_positions, bones_idx=BONES_IDX, markers_per_bone=3,
                            noise_std=0.0, seed=0):
    """
    Sample points along each bone segment [pa, pb].
    Returns:
      markers: (K,3) array,
      marker_bones: list of K (ja, jb) pairs indicating which bone each marker belongs to.
    """
    rng = np.random.default_rng(seed)
    markers = []
    marker_bones = []

    # Avoid exact endpoints to keep markers "on bone" but not at joints
    if markers_per_bone == 1:
        ts = np.array([0.5])
    else:
        ts = np.linspace(0.15, 0.85, markers_per_bone)

    for (ja, jb) in bones_idx:
        pa = joint_positions[ja]
        pb = joint_positions[jb]
        v = pb - pa
        L = np.linalg.norm(v)
        if L < 1e-9:
            continue
        u = v / L
        n1, n2 = _perp_basis(u)

        for t in ts:
            p = pa + t * v
            if noise_std > 0.0:
                p = p + rng.normal(0.0, noise_std) * n1 + rng.normal(0.0, noise_std) * n2
            markers.append(p)
            marker_bones.append((ja, jb))

    return np.asarray(markers), marker_bones

def plot_skeleton_with_markers(theta, bone_lengths=BONE_LENGTHS, markers=None, title='GT with markers'):
    bl = bone_lengths.copy()
    jp, jo = get_joint_positions_and_orientations(bl, theta)

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the skeleton (no targets to keep legend clean)
    plot_skeleton(ax, jp, jo, targets=None, show_axes=True, title=title)

    # Overlay markers
    if markers is not None and len(markers) > 0:
        ax.scatter(markers[:, 0], markers[:, 1], markers[:, 2],
                   marker='x', s=40, label='Markers')
        # ax.legend(loc='upper right')

    plt.tight_layout()
    # plt.show()

    return jp, jo

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
    targets=None, target_names=None,
    markers=None, marker_bones=None, show_projections=True,   # NEW
    show_axes=True, title=''
):
    ax.clear()
    # joints
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], color='red', s=50)

    # optional GT markers
    if markers is not None and len(markers) > 0:
        ax.scatter(markers[:, 0], markers[:, 1], markers[:, 2],
                   marker='x', s=60, label='GT markers')

    # skeleton bones
    for b in BONES_IDX:
        xs, ys, zs = zip(*joint_positions[list(b)])
        ax.plot(xs, ys, zs, color='black', linewidth=2)

    # optional projections from GT markers to fitted bones
    if show_projections and (markers is not None) and (marker_bones is not None):
        overlay_marker_projections(ax, joint_positions, markers, marker_bones, color='C2', alpha=0.6)

    # frames & labels
    for i, (pos, R) in enumerate(zip(joint_positions, joint_orientations)):
        if show_axes:
            draw_frame(ax, pos, R, length=0.05)
        ax.text(pos[0], pos[1], pos[2], f'{i}: {JOINT_NAMES[i]}', color='darkblue', fontsize=9)

    ax.set_xlabel('X (forward)'); ax.set_ylabel('Y (left)'); ax.set_zlabel('Z (up)')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)

    # legend if anything is labeled
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper right')


if __name__ == "__main__":
  
    # 1) Define GT angles
    theta_gt = make_gt_angles()

    # 2) Run FK to get joint positions
    bl = BONE_LENGTHS.copy()
    jp_gt, _ = get_joint_positions_and_orientations(bl, theta_gt)

    # 3) Generate markers along each bone (set noise_std>0 for realism)
    markers_gt, _ = sample_markers_on_bones(jp_gt, BONES_IDX,
                                                       markers_per_bone=3,
                                                       noise_std=0.0,  # e.g., 0.005 for 5 mm noise if units are meters
                                                       seed=42)

    visualize_gt_markers = False  # Set True to see GT pose with markers
    if visualize_gt_markers:
        plot_skeleton_with_markers(theta_gt, bl, markers=markers_gt, title='GT Pose + Bone Markers')

    joint_angles = get_default_joint_angles()

    # 1) Joint limits
    lower_lim, upper_lim = get_default_joint_limits()
    lower_lim, upper_lim = enforce_pure_hinges_in_limits(lower_lim, upper_lim, tight_deg=0.5)
    active_idx = make_active_dof_indices_human_like_hinges()
    
    # Optional semantic priors: e.g., first N markers likely upper body
    #  - Allow region tags: 'upper_body', 'lower_body', 'left_arm', 'right_leg', etc.
    semantic_priors = {
        # 0: ['upper_body'],   # marker 0 restricted to torso/head/arms
        # 10: ['right_leg'],   # marker 10 restricted to right leg bones
    }

    start_time = time.time()

    markers = markers_gt
    marker_weights=np.ones(len(markers))


    joint_angles_ik, bone_lengths_ik, angles_history, bone_length_history = lm_fit_markers_to_bones(
        BONE_LENGTHS, get_default_joint_angles(),
        markers_gt, marker_bones=None,                      # unknown correspondences
        opt_joint_indices_list=[active_idx],
        use_segment=True,
        optimize_bones=True,
        max_iters=200,
        tolerance=1e-3,
        angle_delta=1e-3,
        length_delta=1e-3,
        lm_lambda0=1e-2,
        lm_lambda_factor=2.0,
        angle_step_clip=np.deg2rad(10.0),
        angle_reg=1.0,
        bone_reg=5.0,
        marker_weights=np.ones(len(markers_gt)),
        joint_limits=(lower_lim, upper_lim),
        verbose=True,
        # --- Robust assignment toggles ---
        auto_assign_bones=True,                   # turn on the robust assignment
        assign_topk=3,                            # Top-K soft (set to 1 for hard)
        assign_soft_sigma_factor=0.12,            # broader/softer weighting
        assign_enable_gate=True,
        assign_distance_gate_abs=None,            # or e.g. 0.25
        assign_distance_gate_factor=1.0,          # 1.0 * body scale
        assign_enable_hysteresis=True,
        assign_hysteresis_margin=0.10,            # 10% tolerance to keep previous
        assign_enable_temporal_smoothing=True,
        assign_temporal_smoothing=0.2,            # blend 20% with previous
        assign_semantic_priors=semantic_priors
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
    visualize_ik_iterations = True # Set True to see animation
    jp_final, jo_final = positions_history[-1], orientations_history[-1]
    # overlay_marker_projections(ax, jp_final, markers, marker_bones)
    targets = None
    target_names = None

    # keep state for smoother correspondences across frames (optional)
    viz_state = {'hard': None, 'weights': None}
    
    
    def animate(i):
        jp_i, jo_i = positions_history[i], orientations_history[i]
        # recompute soft correspondences for the current frame (for plotting only)
        corr = robust_assign_markers(
            markers_gt, jp_i, BONES_IDX, use_segment=True, prev_state=viz_state,
            bone_lengths=bone_lengths_ik,
            topk=3, soft_sigma_factor=0.12,
            distance_gate_abs=None, distance_gate_factor=1.0, enable_gate=True,
            hysteresis_margin=0.10, enable_hysteresis=True,
            temporal_smoothing=0.2, enable_temporal_smoothing=True,
            semantic_priors=semantic_priors
        )
        viz_state.update(corr['state'])

        plot_skeleton(
            ax,
            jp_i, jo_i,
            targets=targets,
            target_names=target_names,
            markers=markers_gt,
            # If you prefer hard projections: marker_bones=corr['hard']
            # For soft projections:
            marker_bones=None, show_projections=False,  # avoid double lines
            show_axes=True,
            title=f'IK Iteration {i+1}/{len(positions_history)}'
        )
        if corr['mode'] == 'soft':
            overlay_marker_soft_projections(ax, jp_i, markers_gt, BONES_IDX, corr['cands'], corr['weights'],
                                            use_segment=True, color='C2', alpha=0.8)
        else:
            overlay_marker_projections(ax, jp_i, markers_gt, corr['hard'], color='C2', alpha=0.8)


    if visualize_ik_iterations:
        ani = FuncAnimation(fig, animate, frames=len(positions_history), interval=300, repeat=False)
    else:
        plot_skeleton(
            ax,
            positions_history[-1],
            orientations_history[-1],
            targets=targets,
            target_names=target_names,
            markers=markers,                 # NEW
            marker_bones=None,       # NEW
            show_projections=True,           # NEW
            show_axes=True,
            title='3D Human Skeleton Visualization'
        )
        plt.tight_layout()
    plt.show()
