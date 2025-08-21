from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
from model import SkeletonModel


def _deg(a: float) -> float:
  return np.deg2rad(a)


# ---- angles ---------------------------------------------------------------

def angles_walk(
  skel: SkeletonModel, t: float, f: float = 1.2,
  A_hip_deg: float = 30, A_knee_deg: float = 45,
  A_sh_yaw_deg: float = 25, A_sh_roll_deg: float = 15,
  A_elbow_deg: float = 25
) -> np.ndarray:
  w = 2.0*np.pi*f
  th = skel.default_joint_angles()
  J = skel.JOINT_IDX; DOF = lambda j, a: 3*j+a

  # torso/neck micro
  th[DOF(J['spine_top'],0)] = _deg(5.0) * np.sin(0.5*w*t)
  th[DOF(J['spine_top'],1)] = _deg(3.0) * np.sin(w*t + np.pi/2)
  th[DOF(J['spine_top'],2)] = _deg(2.0) * np.sin(w*t)
  th[DOF(J['neck_top'],1)]  = _deg(2.0) * np.sin(w*t + np.pi/2)
  th[DOF(J['neck_top'],2)]  = _deg(2.0) * np.sin(w*t)

  # legs
  A_hip, A_knee = _deg(A_hip_deg), _deg(A_knee_deg)
  th[DOF(J['right_hip'],1)]  = A_hip*np.sin(w*t)
  th[DOF(J['left_hip'],1)]   = -A_hip*np.sin(w*t)
  th[DOF(J['right_knee'],1)] = 0.5*A_knee*(1.0 - np.cos(w*t))
  th[DOF(J['left_knee'],1)]  = 0.5*A_knee*(1.0 - np.cos(w*t + np.pi))

  # arms
  A_y, A_r, A_e = _deg(A_sh_yaw_deg), _deg(A_sh_roll_deg), _deg(A_elbow_deg)
  th[DOF(J['right_shoulder'],0)] = A_y*np.sin(w*t + np.pi)
  th[DOF(J['left_shoulder'],0)]  = A_y*np.sin(w*t)
  th[DOF(J['right_shoulder'],2)] = -A_r*np.sin(w*t + np.pi)
  th[DOF(J['left_shoulder'],2)]  = A_r*np.sin(w*t)
  th[DOF(J['right_elbow'],1)]    = 0.5*A_e*(1.0 - np.cos(w*t + np.pi))
  th[DOF(J['left_elbow'],1)]     = 0.5*A_e*(1.0 - np.cos(w*t))

  return th


def angles_run(
  skel: SkeletonModel, t: float, f: float = 2.4,
  A_hip_deg: float = 45, A_knee_deg: float = 75,
  A_sh_yaw_deg: float = 40, A_elbow_deg: float = 40,
  pelvis_pitch_deg: float = 5
) -> np.ndarray:
  w = 2*np.pi*f
  th = skel.default_joint_angles()
  J = skel.JOINT_IDX; DOF = lambda j, a: 3*j+a

  th[DOF(J['pelvis'],1)] = _deg(pelvis_pitch_deg)

  # legs
  A_hip, A_knee = _deg(A_hip_deg), _deg(A_knee_deg)
  th[DOF(J['right_hip'],1)]  = A_hip*np.sin(w*t)
  th[DOF(J['left_hip'],1)]   = -A_hip*np.sin(w*t)
  th[DOF(J['right_knee'],1)] = 0.5*A_knee*(1.0 - np.cos(w*t))
  th[DOF(J['left_knee'],1)]  = 0.5*A_knee*(1.0 - np.cos(w*t + np.pi))

  # arms
  A_y, A_e = _deg(A_sh_yaw_deg), _deg(A_elbow_deg)
  th[DOF(J['right_shoulder'],0)] = A_y*np.sin(w*t + np.pi)
  th[DOF(J['left_shoulder'],0)]  = A_y*np.sin(w*t)
  th[DOF(J['right_elbow'],1)]    = 0.6*A_e*(1.0 - np.cos(w*t + np.pi))
  th[DOF(J['left_elbow'],1)]     = 0.6*A_e*(1.0 - np.cos(w*t))

  return th


def angles_turn_in_place(
  skel: SkeletonModel, t: float, turn_rate_deg_s: float = 60
) -> np.ndarray:
  th = skel.default_joint_angles()
  J = skel.JOINT_IDX; DOF = lambda j, a: 3*j+a
  th[DOF(J['pelvis'],0)] = _deg(turn_rate_deg_s) * t
  return th


# ---- root translations ---------------------------------------------------

def root_walk(t: float, speed: float = 1.0, f: float = 1.2) -> np.ndarray:
  w = 2.0*np.pi*f
  return np.array([speed*t, 0.03*np.sin(w*t+np.pi/2), 0.02*np.sin(2*w*t)], float)


def root_run(t: float, speed: float = 3.0, f: float = 2.4) -> np.ndarray:
  w = 2.0*np.pi*f
  return np.array([speed*t, 0.04*np.sin(w*t+np.pi/2),
                   0.05*np.maximum(0.0, np.sin(2*w*t))], float)


def root_turn_in_place(t: float) -> np.ndarray:
  return np.zeros(3, float)


# ---- factory -------------------------------------------------------------

def make_motion(
  skel: SkeletonModel, kind: str, **kw
) -> Tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
  kind = kind.lower()
  if kind == "walk":
    f = kw.get("f", 1.2)
    speed = kw.get("speed", 1.0)
    return (
      lambda t: angles_walk(
        skel, t, f=f,
        A_hip_deg=kw.get("A_hip_deg",30),
        A_knee_deg=kw.get("A_knee_deg",45),
        A_sh_yaw_deg=kw.get("A_sh_yaw_deg",25),
        A_sh_roll_deg=kw.get("A_sh_roll_deg",15),
        A_elbow_deg=kw.get("A_elbow_deg",25)
      ),
      lambda t: root_walk(t, speed=speed, f=f)
    )

  if kind == "run":
    f = kw.get("f", 2.4)
    speed = kw.get("speed", 3.0)
    return (
      lambda t: angles_run(
        skel, t, f=f,
        A_hip_deg=kw.get("A_hip_deg",45),
        A_knee_deg=kw.get("A_knee_deg",75),
        A_sh_yaw_deg=kw.get("A_sh_yaw_deg",40),
        A_elbow_deg=kw.get("A_elbow_deg",40),
        pelvis_pitch_deg=kw.get("pelvis_pitch_deg",5)
      ),
      lambda t: root_run(t, speed=speed, f=f)
    )

  if kind == "turn":
    return (
      lambda t: angles_turn_in_place(
        skel, t, turn_rate_deg_s=kw.get("turn_rate_deg_s",60)
      ),
      lambda t: root_turn_in_place(t)
    )

  raise ValueError(f"Unknown motion kind: {kind}. Try one of: walk, run, turn.")
