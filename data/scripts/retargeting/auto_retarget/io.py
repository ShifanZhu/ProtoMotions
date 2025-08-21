from __future__ import annotations
from typing import List, Tuple, Sequence
import numpy as np
from model import SkeletonModel


class DataIO:
  """CSV readers for AMASS-style and generic marker files."""

  @staticmethod
  def load_amass_markers_file(
    path: str,
    skel: SkeletonModel,
    *,
    axis_permutation: Sequence[int] = (0, 1, 2),
    axis_flip: Sequence[float] = (1.0, 1.0, 1.0),
  ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Read AMASS-like CSV and map to this skeleton's bone pairs."""
    J = skel.JOINT_IDX
    AMASS_TO_OUR = [
      ("Pelvis", (J['pelvis'], J['spine_top'])),
      ("Chest", (J['spine_top'], J['neck_top'])),
      ("Neck", (J['spine_top'], J['neck_top'])),
      ("Head", (J['neck_top'], J['head_top'])),
      ("R_Shoulder", (J['spine_top'], J['right_shoulder'])),
      ("R_Elbow", (J['right_shoulder'], J['right_elbow'])),
      ("R_Wrist", (J['right_elbow'], J['right_hand'])),
      ("L_Shoulder", (J['spine_top'], J['left_shoulder'])),
      ("L_Elbow", (J['left_shoulder'], J['left_elbow'])),
      ("L_Wrist", (J['left_elbow'], J['left_hand'])),
      ("R_Hip", (J['pelvis'], J['right_hip'])),
      ("R_Knee", (J['right_hip'], J['right_knee'])),
      ("R_Ankle", (J['right_knee'], J['right_foot'])),
      ("L_Hip", (J['pelvis'], J['left_hip'])),
      ("L_Knee", (J['left_hip'], J['left_knee'])),
      ("L_Ankle", (J['left_knee'], J['left_foot'])),
    ]

    rec = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None, autostrip=True)
    if rec.shape == ():
      rec = rec.reshape(1)
    cols = rec.dtype.names

    def _have(base: str) -> bool:
      return (f"{base}_x" in cols) and (f"{base}_y" in cols) and (f"{base}_z" in cols)

    missing = [base for base, _ in AMASS_TO_OUR if not _have(base)]
    if missing:
      raise ValueError(f"AMASS file is missing columns for: {missing}")

    T = rec.shape[0]
    K = len(AMASS_TO_OUR)
    markers = np.zeros((T, K, 3), float)

    perm = np.array(axis_permutation, int)
    flip = np.array(axis_flip, float).reshape(1, 3)

    for k, (base, _) in enumerate(AMASS_TO_OUR):
      xyz = np.stack([rec[f"{base}_x"], rec[f"{base}_y"], rec[f"{base}_z"]], axis=1)
      xyz = xyz[:, perm] * flip
      markers[:, k, :] = xyz

    times = np.asarray(rec["time_s"], float) if "time_s" in cols else np.arange(T, float) / 30.0
    marker_bones = [seg for _, seg in AMASS_TO_OUR]
    return times, markers, marker_bones

  @staticmethod
  def load_markers_csv_generic(
    path: str,
    *,
    axis_permutation: Sequence[int] = (0, 1, 2),
    axis_flip: Sequence[float] = (1.0, 1.0, 1.0),
  ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read any CSV with <name>_{x,y,z} columns."""
    rec = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None, autostrip=True)
    if rec.shape == ():
      rec = rec.reshape(1)
    cols = rec.dtype.names

    bases: List[str] = []
    seen = set()
    for c in cols:
      if c.endswith("_x"):
        base = c[:-2]
        if (f"{base}_y" in cols) and (f"{base}_z" in cols) and base not in seen:
          bases.append(base)
          seen.add(base)

    if not bases:
      raise ValueError("No <name>_{x,y,z} triplets found in header.")

    T = rec.shape[0]
    K = len(bases)
    markers = np.zeros((T, K, 3), float)
    perm = np.array(axis_permutation, int)
    flip = np.array(axis_flip, float).reshape(1, 3)

    for k, base in enumerate(bases):
      xyz = np.stack([rec[f"{base}_x"], rec[f"{base}_y"], rec[f"{base}_z"]], axis=1)
      xyz = xyz[:, perm] * flip
      markers[:, k, :] = xyz

    times = np.asarray(rec["time_s"], float) if "time_s" in cols else np.arange(T, float) / 30.0
    return times, markers, bases
