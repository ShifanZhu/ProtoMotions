from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .model import SkeletonModel


def make_active_dof_indices_human_like_hinges(skel: SkeletonModel) -> List[int]:
  """Allow torso/neck (subset), shoulders/hips (3-DOF) + elbows/knees pitch-only."""
  J = skel.JOINT_IDX
  DOF = lambda j, a: 3 * j + a
  active: List[int] = []

  # torso/neck
  active += [
    DOF(J['spine_top'], 0), DOF(J['spine_top'], 1), DOF(J['spine_top'], 2),
    DOF(J['neck_top'], 1), DOF(J['neck_top'], 2)
  ]

  # shoulders + elbows
  active += [
    DOF(J['right_shoulder'], 0), DOF(J['right_shoulder'], 2), DOF(J['right_elbow'], 0),
    DOF(J['left_shoulder'], 0), DOF(J['left_shoulder'], 2), DOF(J['left_elbow'], 0)
  ]

  # hips
  active += [
    DOF(J['right_hip'], 0), DOF(J['right_hip'], 1), DOF(J['right_hip'], 2),
    DOF(J['left_hip'], 0), DOF(J['left_hip'], 1), DOF(J['left_hip'], 2)
  ]

  # hinge-only pitch
  active += [
    DOF(J['right_elbow'], 1), DOF(J['left_elbow'], 1),
    DOF(J['right_knee'], 1), DOF(J['left_knee'], 1)
  ]

  return sorted(set(active))


def enforce_pure_hinges_in_limits(
  lower: np.ndarray,
  upper: np.ndarray,
  skel: SkeletonModel,
  tight_deg: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
  """Zero-out yaw/roll at elbows/knees to create pure hinges (±tight_deg)."""
  eps = np.deg2rad(tight_deg)
  J = skel.JOINT_IDX

  for jn in ("right_elbow", "left_elbow", "right_knee", "left_knee"):
    i = 3 * J[jn]
    lower[i + 0], upper[i + 0] = -eps, eps   # yaw
    lower[i + 2], upper[i + 2] = -eps, eps   # roll

  return lower, upper


class ConstraintsManager:
  """Helper to apply standard human-like hinge constraints."""
  def __init__(self, skel: SkeletonModel):
    self.skel = skel

  def active_dofs(self) -> List[int]:
    return make_active_dof_indices_human_like_hinges(self.skel)

  def enforce_limits(
    self, lower: np.ndarray, upper: np.ndarray, tight_deg: float = 0.5
  ) -> Tuple[np.ndarray, np.ndarray]:
    return enforce_pure_hinges_in_limits(lower, upper, self.skel, tight_deg)
