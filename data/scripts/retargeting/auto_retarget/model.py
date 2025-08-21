from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


class SkeletonModel:
  """Robot/skeleton topology, defaults, and forward kinematics (FK)."""

  # ---- Topology ---------------------------------------------------------
  JOINT_NAMES: List[str] = [
    "pelvis", "spine_top", "neck_top", "head_top",
    "right_shoulder", "right_elbow", "right_hand",
    "left_shoulder", "left_elbow", "left_hand",
    "right_hip", "right_knee", "right_foot",
    "left_hip", "left_knee", "left_foot",
  ]
  JOINT_IDX: Dict[str, int] = {n: i for i, n in enumerate(JOINT_NAMES)}

  # Expose indices as class attributes for convenience
  (
    PELVIS, SPINE_TOP, NECK_TOP, HEAD_TOP,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HAND,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_HAND,
    RIGHT_HIP, RIGHT_KNEE, RIGHT_FOOT,
    LEFT_HIP, LEFT_KNEE, LEFT_FOOT,
  ) = range(16)

  BONES_IDX: List[Tuple[int, int]] = [
    (PELVIS, SPINE_TOP), (SPINE_TOP, NECK_TOP), (NECK_TOP, HEAD_TOP),
    (SPINE_TOP, RIGHT_SHOULDER), (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_HAND),
    (SPINE_TOP, LEFT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_HAND),
    (PELVIS, RIGHT_HIP), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_FOOT),
    (PELVIS, LEFT_HIP), (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_FOOT),
  ]
  BONE_PAIR_TO_INDEX: Dict[Tuple[int, int], int] = {p: i for i, p in enumerate(BONES_IDX)}

  # ---- Defaults ---------------------------------------------------------
  BONE_LENGTHS: Dict[str, float] = {
    'spine': 0.5, 'neck': 0.1, 'head': 0.1,
    'upper_arm': 0.3, 'lower_arm': 0.25,
    'upper_leg': 0.4, 'lower_leg': 0.4,
    'shoulder_offset': 0.2, 'hip_offset': 0.1,
  }

  BONE_THICKNESS: Dict[str, float] = {
    'spine': 0.10, 'neck': 0.06, 'head': 0.09,
    'upper_arm': 0.045, 'lower_arm': 0.040,
    'upper_leg': 0.070, 'lower_leg': 0.060,
    'shoulder_offset': 0.060, 'hip_offset': 0.065,
  }

  BONE_LENGTH_KEYS_TO_OPTIMIZE = [
    'upper_arm', 'lower_arm', 'upper_leg', 'lower_leg', 'neck'
  ]

  # ---- Small helpers ----------------------------------------------------
  @staticmethod
  def DOF(joint: int, axis: int) -> int:
    return 3 * joint + axis

  @staticmethod
  def deg(a: float) -> float:
    return np.deg2rad(a)
