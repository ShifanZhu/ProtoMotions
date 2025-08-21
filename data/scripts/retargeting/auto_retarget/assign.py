from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from model import SkeletonModel as SM
from geometry import Geometry3D as G


@dataclass
class AssignConfig:
  topk: int = 1
  soft_sigma_factor: float = 0.1
  enable_gate: bool = True
  distance_gate_abs: Optional[float] = None
  distance_gate_factor: float = 1.0
  enable_hysteresis: bool = True
  hysteresis_margin: float = 0.10
  enable_temporal_smoothing: bool = False
  temporal_smoothing: float = 0.0
  semantic_priors: Optional[Dict] = None
  geom: str = "segment"  # "line" | "segment" | "capsule" | "cylinder"
  bone_radii: Optional[np.ndarray] = None


class MarkerAssigner:
  """Robust marker→bone correspondence with optional soft assignments."""

  def __init__(self, model: SM):
    self.model = model

  def assign(
    self,
    markers: np.ndarray,
    joint_positions: np.ndarray,
    *,
    bones_idx: Sequence[Tuple[int, int]] | None = None,
    prev_state: Optional[Dict] = None,
    bone_lengths: Optional[Dict[str, float]] = None,
    cfg: AssignConfig = AssignConfig(),
  ):
    if bones_idx is None:
      bones_idx = SM.BONES_IDX
    n_bones = len(bones_idx)

    # scale from bone lengths
    scale = self.model.compute_scale(bone_lengths or self.model.BONE_LENGTHS)
    sigma = max(1e-6, cfg.soft_sigma_factor * scale)
    gate = (
      cfg.distance_gate_abs
      if (cfg.enable_gate and cfg.distance_gate_abs is not None)
      else (cfg.distance_gate_factor * scale if cfg.enable_gate else np.inf)
    )

    allowed_by_marker = self.model.expand_semantic_priors(cfg.semantic_priors, bones_idx)

    prev_hard = prev_state.get("hard") if prev_state else None
    prev_wts = prev_state.get("weights") if prev_state else None

    cands: List[List[int]] = []
    weights: List[np.ndarray] = []
    hard_pairs: List[Tuple[int, int]] = []

    for mi, m in enumerate(np.asarray(markers)):
      allowed = allowed_by_marker.get(mi)
      dists: List[float] = []
      idxs: List[int] = []

      for bi, (ja, jb) in enumerate(bones_idx):
        if (allowed is not None) and (bi not in allowed):
          continue

        a, b = joint_positions[ja], joint_positions[jb]

        if cfg.geom == "segment":
          # TODO: implement distance calc against segment
          pass
        elif cfg.geom == "line":
          # TODO: implement distance calc against line
          pass
        elif cfg.geom in ("capsule", "cylinder"):
          # TODO: implement capsule/cylinder
          pass
        else:
          raise ValueError(f"Unsupported geom {cfg.geom}")

      # TODO: fill cands, weights, hard_pairs with results

    return {
      "candidates": cands,
      "weights": weights,
      "hard": hard_pairs,
    }
