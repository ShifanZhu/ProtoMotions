#!/usr/bin/env python3
"""
Demo: Fit skeleton to synthetic marker data.
"""

import numpy as np
from model import SkeletonModel
from optim import LMOptimizer
from plot import Plotter
from motions import make_synthetic_motion

def main():
  # 1. Generate synthetic ground-truth motion and markers
  skel = SkeletonModel()
  T = 100
  gt_thetas, gt_roots, markers_seq = make_synthetic_motion(skel, T)

  # 2. Init optimizer and plotter
  optimizer = LMOptimizer(skel)
  plotter = Plotter(skel)

  # 3. Fit each frame
  theta = np.zeros(skel.n_dofs)
  root = np.zeros(3)
  bone_lengths = skel.BONE_LENGTHS

  for fidx, markers in enumerate(markers_seq):
    theta, root = optimizer.fit_frame(
      markers=markers,
      theta_init=theta,
      root_init=root,
      bone_lengths=bone_lengths,
    )

    err = np.linalg.norm(theta - gt_thetas[fidx])
    if fidx % 10 == 0:
      print(f"[Frame {fidx}] Joint angle error = {err:.4f}")

    plotter.update(markers, theta, root, gt_pose=(gt_thetas[fidx], gt_roots[fidx]))

  plotter.show()

if __name__ == "__main__":
  main()
