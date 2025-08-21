from __future__ import annotations
import numpy as np
from typing import List, Tuple, Sequence
from model import SkeletonModel as SM


class Geometry3D:
  """Distance, closest-point, and residual builders (vectorized support)."""

  # ---- Perpendicular basis ---------------------------------------------
  @staticmethod
  def perp_basis(u: np.ndarray):
    u = u / (np.linalg.norm(u) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n1 = np.cross(u, tmp); n1 /= (np.linalg.norm(n1) + 1e-12)
    n2 = np.cross(u, n1); n2 /= (np.linalg.norm(n2) + 1e-12)
    return n1, n2

  @staticmethod
  def perp_basis_vec(u: np.ndarray):
    eps = 1e-12
    mask = np.abs(u[:, 0]) < 0.9
    tmp = np.empty_like(u)
    tmp[mask] = np.array([1.0, 0.0, 0.0])
    tmp[~mask] = np.array([0.0, 1.0, 0.0])
    n1 = np.cross(u, tmp)
    n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + eps)
    n2 = np.cross(u, n1)
    n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + eps)
    return n1, n2

  # ---- Closest points ---------------------------------------------------
  @staticmethod
  def _closest_point_on_segment_pointwise(p, a, b):
    v = b - a
    L2 = float(v @ v)
    if L2 < 1e-12:
      return a
    t = float((p - a) @ v) / L2
    t = np.clip(t, 0.0, 1.0)
    return a + t * v

  @classmethod
  def closest_point_on_segment(cls, p, a, b):
    return cls._closest_point_on_segment_pointwise(p, a, b)

  @staticmethod
  def closest_point_on_line(p, a, b):
    v = b - a
    L2 = float(v @ v)
    if L2 < 1e-12:
      return a
    t = float((p - a) @ v) / L2
    return a + t * v

  @classmethod
  def closest_point_on_capsule_surface(cls, p, a, b, R):
    q = cls._closest_point_on_segment_pointwise(p, a, b)
    d = p - q
    return q + d / (np.linalg.norm(d) + 1e-12) * R
