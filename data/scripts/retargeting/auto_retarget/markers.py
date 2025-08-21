from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from model import SkeletonModel as SM
from geometry import Geometry3D as G


class MarkerTemplate:
  """Per-bone (t, phi) sampling template and renderer."""

  @staticmethod
  def make_template(bones_idx=None, *, markers_per_bone=8, geom="cylinder", seed=0):
    if bones_idx is None:
      bones_idx = SM.BONES_IDX
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.05, 0.95, markers_per_bone)
    template = []
    for _bi, _pair in enumerate(bones_idx):
      rot_off = rng.random() * 2.0 * np.pi
      bone_entries = []
      for k, t in enumerate(ts):
        phi = (2.0 * np.pi * k / markers_per_bone) + rot_off
        bone_entries.append((float(t), float(phi)))
      template.append(bone_entries)
    return {"geom": geom, "entries": template, "markers_per_bone": markers_per_bone}

  @staticmethod
  def render(
    jp: np.ndarray,
    template,
    *,
    bone_radii=None,
    jitter_tangent_std=3.0,
    seed=0,
    bones_idx=None
  ):
    if bones_idx is None:
      bones_idx = SM.BONES_IDX
    rng = np.random.default_rng(seed)
    geom = template["geom"]
    entries = template["entries"]
    markers = []
    marker_bones = []
    for bi, (ja, jb) in enumerate(bones_idx):
      a, b = jp[ja], jp[jb]
      v = b - a
      L = np.linalg.norm(v)
      if L < 1e-9:
        continue
      u = v / L
      n1, n2 = G.perp_basis(u)
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
          p = (G.closest_point_on_capped_cylinder_surface(p, a, b, R)
               if geom == "cylinder" else
               G.closest_point_on_capsule_surface(p, a, b, R))
        else:
          raise ValueError("geom must be 'segment', 'cylinder', or 'capsule'")
        markers.append(p)
        marker_bones.append((ja, jb))
    return np.asarray(markers), marker_bones
