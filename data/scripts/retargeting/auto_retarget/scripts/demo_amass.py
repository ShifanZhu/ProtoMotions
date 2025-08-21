#!/usr/bin/env python3
"""
Demo: Fit skeleton model to AMASS motion capture data.

Usage:
    python demo_amass.py --file <path/to/amass.npy>
"""

import argparse
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from markers import MarkerAssigner, AssignConfig
from model import SkeletonModel
from optim import LMOptimizer
from plot import Plotter
from io import load_amass

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--file", type=str, required=True, help="Path to AMASS npz or npy file")
  args = parser.parse_args()

  # 1. Load AMASS markers
  markers_seq = load_amass(args.file)  # shape (T, K, 3)

  # 2. Init skeleton model
  skel = SkeletonModel()
  optimizer = LMOptimizer(skel)
  plotter = Plotter(skel)

  # 3. Loop through frames
  theta = np.zeros(skel.n_dofs)  # start with neutral pose
  bone_lengths = skel.BONE_LENGTHS
  root = np.zeros(3)

  for fidx, markers in enumerate(markers_seq):
    theta, root = optimizer.fit_frame(
      markers=markers,
      theta_init=theta,
      root_init=root,
      bone_lengths=bone_lengths,
    )

    if fidx % 10 == 0:
      print(f"[Frame {fidx}] Optimized pose error: {optimizer.last_error:.4f}")

    plotter.update(markers, theta, root)
  
  plotter.show()

if __name__ == "__main__":
  main()
