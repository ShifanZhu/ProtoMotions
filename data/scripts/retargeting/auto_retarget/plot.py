from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib as mpl




class Plotter3D:
  """All drawing utilities for the skeleton and markers."""


  # ---- low-level -------------------------------------------------------
  @staticmethod
  def draw_frame(ax, origin: np.ndarray, R: np.ndarray, length: float = 0.05):
    x_axis = origin + R @ np.array([length, 0, 0])
    y_axis = origin + R @ np.array([0, length, 0])
    z_axis = origin + R @ np.array([0, 0, length])
    ax.plot([origin[0], x_axis[0]],[origin[1], x_axis[1]],[origin[2], x_axis[2]], color='r', linewidth=2)
    ax.plot([origin[0], y_axis[0]],[origin[1], y_axis[1]],[origin[2], y_axis[2]], color='g', linewidth=2)
    ax.plot([origin[0], z_axis[0]],[origin[1], z_axis[1]],[origin[2], z_axis[2]], color='b', linewidth=2)


  @staticmethod
  def set_axes_equal(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle = np.mean(x_limits); y_middle = np.mean(y_limits); z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


  @staticmethod
  def _perp_basis(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u = u / (np.linalg.norm(u) + 1e-12)
    tmp = np.array([1.0,0.0,0.0]) if abs(u[0]) < 0.9 else np.array([0.0,1.0,0.0])
    n1 = np.cross(u, tmp); n1 /= (np.linalg.norm(n1) + 1e-12)
    n2 = np.cross(u, n1); n2 /= (np.linalg.norm(n2) + 1e-12)
    return n1, n2


  @classmethod
  def draw_cylinder(cls, ax, a: np.ndarray, b: np.ndarray, R: float, color=(0.75,0.75,0.8), alpha: float = 0.25, n_theta: int = 18, n_len: int = 6):
    v = b - a
    L = np.linalg.norm(v)
    if L < 1e-9:
      return
    u = v / L
    n1, n2 = cls._perp_basis(u)
    t_vals = np.linspace(0.0, L, n_len)
    th = np.linspace(0, 2*np.pi, n_theta)
    ct, st = np.cos(th), np.sin(th)
    X = []; Y = []; Z = []
    for t in t_vals:
      c = a + t * u
      ring = np.outer(ct, n1) + np.outer(st, n2)
      pts = c + R * ring
      X.append(pts[:,0]); Y.append(pts[:,1]); Z.append(pts[:,2])
    X = np.vstack(X); Y = np.vstack(Y); Z = np.vstack(Z)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, color=color, alpha=alpha, shade=True)


  @staticmethod
  def draw_skeleton_wire(ax, joint_positions: np.ndarray, bones_idx: List[Tuple[int,int]], color='k', lw: float = 2, alpha: float = 1.0):
    for (ja, jb) in bones_idx:
      xs, ys, zs = zip(*joint_positions[[ja, jb]])
      ax.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha)


  @classmethod
  def plot_skeleton(
    cls, ax,
    joint_positions: np.ndarray,
    bones_idx: List[Tuple[int, int]],
    *,
    markers: Optional[np.ndarray] = None,
    joint_color: str = 'red',
    wire_color: str = 'black',
    wire_alpha: float = 1.0,
    show_axes: bool = False,
    draw_solids: bool = False,
    bone_radii: Optional[np.ndarray] = None,
    title: str = '',
    clear: bool = True,
  ):
    if clear: ax.clear()
    ax.scatter(joint_positions[:,0], joint_positions[:,1], joint_positions[:,2], color=joint_color, s=35, alpha=wire_alpha)
    cls.draw_skeleton_wire(ax, joint_positions, bones_idx, color=wire_color, lw=2, alpha=wire_alpha)
    if draw_solids and bone_radii is not None:
      for bi, (ja, jb) in enumerate(bones_idx):
        R = float(bone_radii[bi])
        cls.draw_cylinder(ax, joint_positions[ja], joint_positions[jb], R, alpha=0.2)
    if markers is not None and len(markers) > 0:
      ax.scatter(markers[:,0], markers[:,1], markers[:,2], marker='x', s=35, color='C1')
    ax.set_xlabel('X (forward)'); ax.set_ylabel('Y (left)'); ax.set_zlabel('Z (up)')
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    Plotter3D.set_axes_equal(ax)


@staticmethod
def save_animation(ani: FuncAnimation, filename_base: str, fps: int = 30, dpi: int = 150):
  try:
    try:
      import imageio_ffmpeg
      mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
      pass
    writer = FFMpegWriter(fps=fps, metadata={'artist': 'skeletonfit'}, bitrate=1800)
    ani.save(f"{filename_base}.mp4", writer=writer, dpi=dpi)
    print(f"Saved {filename_base}.mp4 with FFmpeg")
    return
  except Exception as e:
    print(f"FFmpeg not available or failed ({e}). Falling back to GIF...")
  try:
    writer = PillowWriter(fps=fps)
    ani.save(f"{filename_base}.gif", writer=writer, dpi=dpi)
    print(f"Saved {filename_base}.gif with PillowWriter")
  except Exception as e:
    print(f"PillowWriter also failed: {e}")
    print("As a last resort, save frames and stitch externally.")