#!/usr/bin/env python3
"""
Play back a marker sequence from data/output/test.markers.txt and optionally save a video.

Supported input formats
-----------------------
1) LONG format (one row per marker per frame):
   frame, time_s, name, x, y, z

2) WIDE format (one row per frame):
   frame, time_s, Pelvis_x, Pelvis_y, Pelvis_z, LeftKnee_x, LeftKnee_y, LeftKnee_z, ...

Usage
-----
python play_markers.py data/output/test.markers.txt --fps 30
# Save to mp4 (needs ffmpeg in PATH, otherwise falls back to GIF):
python play_markers.py data/output/test.markers.txt --save out.mp4
# Save GIF:
python play_markers.py data/output/test.markers.txt --save out.gif

"""

import argparse
import os
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def set_axes_equal(ax):
    """Make 3D axes have equal scale so that spheres look like spheres."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def _read_table(path: str) -> pd.DataFrame:
    # Let pandas infer comma, tab, etc.
    return pd.read_csv(path, sep=None, engine="python")

def _detect_format(df: pd.DataFrame) -> str:
    cols = set(c.lower() for c in df.columns)
    if {"frame", "name", "x", "y", "z"}.issubset(cols):
        return "long"
    # look for any *_x, *_y, *_z columns besides frame/time
    pat = re.compile(r"(.+)_([xyz])$", flags=re.IGNORECASE)
    has_xyz = any(pat.match(c) for c in df.columns if c.lower() not in ("frame", "time", "time_s"))
    return "wide" if has_xyz else "unknown"

def _extract_from_long(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], np.ndarray]:
    # normalize column names
    ren = {c: c.lower() for c in df.columns}
    df = df.rename(columns=ren)
    if "time" in df.columns and "time_s" not in df.columns:
        df = df.rename(columns={"time": "time_s"})

    # get marker names in a stable order (alphabetical)
    names = sorted(df["name"].unique().tolist())
    frames = np.sort(df["frame"].unique())

    # times (if present)
    times = []
    if "time_s" in df.columns:
        # one time per frame (take first occurrence)
        times = df.groupby("frame")["time_s"].first().reindex(frames).to_numpy()
    else:
        times = np.arange(len(frames), dtype=float)

    # assemble [F, M, 3]
    M = len(names)
    F = len(frames)
    out = np.zeros((F, M, 3), dtype=float)

    # faster pivot per frame
    name_to_idx = {n: i for i, n in enumerate(names)}
    grouped = df.groupby("frame")
    for fi, fr in enumerate(frames):
        g = grouped.get_group(fr)
        # ensure we only use expected names and keep last occurrence if duplicates
        last = g.drop_duplicates(subset=["name"], keep="last").set_index("name")
        for n, row in last.iterrows():
            if n in name_to_idx:
                mi = name_to_idx[n]
                out[fi, mi, 0] = row["x"]
                out[fi, mi, 1] = row["y"]
                out[fi, mi, 2] = row["z"]
    return out, names, np.array(times)

def _extract_from_wide(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], np.ndarray]:
    ren = {c: c.lower() for c in df.columns}
    df = df.rename(columns=ren)
    if "time" in df.columns and "time_s" not in df.columns:
        df = df.rename(columns={"time": "time_s"})
    if "frame" not in df.columns:
        # create a frame index if missing
        df.insert(0, "frame", np.arange(len(df), dtype=int))

    # get marker base names by grouping *_x,y,z
    pat = re.compile(r"(.+)_([xyz])$")
    base = {}
    for c in df.columns:
        m = pat.match(c)
        if m:
            base.setdefault(m.group(1), set()).add(m.group(2))
    # only accept markers that have all x,y,z
    names = sorted([n for n, axes in base.items() if {"x", "y", "z"}.issubset(axes)])
    frames = df["frame"].to_numpy()
    times = df["time_s"].to_numpy() if "time_s" in df.columns else np.arange(len(frames), dtype=float)

    F, M = len(frames), len(names)
    out = np.zeros((F, M, 3), dtype=float)
    for i, n in enumerate(names):
        out[:, i, 0] = df[f"{n}_x"].to_numpy()
        out[:, i, 1] = df[f"{n}_y"].to_numpy()
        out[:, i, 2] = df[f"{n}_z"].to_numpy()

    return out, names, times

def load_markers(path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    df = _read_table(path)
    fmt = _detect_format(df)
    if fmt == "long":
        return _extract_from_long(df)
    elif fmt == "wide":
        return _extract_from_wide(df)
    else:
        raise ValueError(
            "Unrecognized file format. Expected either LONG format "
            "(frame,time_s,name,x,y,z) or WIDE format (frame,time_s,Name_x,Name_y,Name_z, ...)."
        )

def main():
    ap = argparse.ArgumentParser(description="Play a marker sequence and optionally save video.")
    ap.add_argument("input", type=str, help="Path to markers file (e.g., data/output/test.markers.txt)")
    ap.add_argument("--fps", type=float, default=None, help="Playback FPS (default: use time_s if present, else 30)")
    ap.add_argument("--stride", type=int, default=1, help="Use every N-th frame for speed (default: 1)")
    ap.add_argument("--ms", type=float, default=28, help="Marker size for scatter points")
    ap.add_argument("--save", type=str, default=None, help="Output path (.mp4 or .gif). If omitted, just plays.")
    ap.add_argument("--dpi", type=int, default=140, help="DPI for saved video")
    ap.add_argument("--elev", type=float, default=15.0, help="3D view elevation (deg)")
    ap.add_argument("--azim", type=float, default=-60.0, help="3D view azimuth (deg)")
    ap.add_argument("--title", type=str, default="Markers Playback", help="Figure title prefix")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")

    XYZ, names, times = load_markers(args.input)   # XYZ: [F, M, 3]
    if args.stride > 1:
        XYZ = XYZ[::args.stride]
        times = times[::args.stride]

    F, M, _ = XYZ.shape
    if F == 0 or M == 0:
        raise RuntimeError("No frames or markers found in the file.")

    # derive fps
    if args.fps is not None:
        fps = float(args.fps)
    else:
        # Try to infer fps from time_s
        if len(times) >= 2 and np.all(np.isfinite(times)):
            dt = np.median(np.diff(times))
            fps = 1.0 / dt if dt > 0 else 30.0
        else:
            fps = 30.0

    # set up colors (stable per marker)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(M)]

    # bounding box for consistent camera
    mins = XYZ.reshape(-1, 3).min(axis=0)
    maxs = XYZ.reshape(-1, 3).max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    pad = 0.10 * span

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)

    scat = ax.scatter(XYZ[0, :, 0], XYZ[0, :, 1], XYZ[0, :, 2], s=args.ms, c=colors)
    ttl = ax.set_title(f"{args.title} — frame 1/{F}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(center[0] - 0.5 * span - pad, center[0] + 0.5 * span + pad)
    ax.set_ylim(center[1] - 0.5 * span - pad, center[1] + 0.5 * span + pad)
    ax.set_zlim(center[2] - 0.5 * span - pad, center[2] + 0.5 * span + pad)
    set_axes_equal(ax)

    # (optional) annotate a few markers so you can identify them
    texts = []
    for i, name in enumerate(names[: min(8, M)]):  # annotate up to 8 names
        x, y, z = XYZ[0, i]
        texts.append(ax.text(x, y, z, name, fontsize=8, alpha=0.8))

    def update(i):
        xs, ys, zs = XYZ[i, :, 0], XYZ[i, :, 1], XYZ[i, :, 2]
        # Matplotlib 3D scatter update:
        scat._offsets3d = (xs, ys, zs)
        ttl.set_text(f"{args.title} — frame {i+1}/{F}")
        for j, t in enumerate(texts):
            if j < M:
                t.set_position((xs[j], ys[j]))
                t.set_3d_properties(zs[j], zdir="z")
        return scat, ttl, *texts

    interval_ms = int(round(1000.0 / fps))
    ani = FuncAnimation(fig, update, frames=F, interval=interval_ms, blit=False, repeat=True)

    if args.save:
        base, ext = os.path.splitext(args.save)
        ext = ext.lower()
        try:
            if ext == ".mp4":
                try:
                    from imageio_ffmpeg import get_ffmpeg_exe  # helps on some systems
                    plt.rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()
                except Exception:
                    pass
                writer = FFMpegWriter(fps=fps, metadata={"artist": "markers"}, bitrate=2000)
                ani.save(args.save, writer=writer, dpi=args.dpi)
                print(f"Saved video: {args.save}")
            elif ext == ".gif":
                writer = PillowWriter(fps=fps)
                ani.save(args.save, writer=writer, dpi=args.dpi)
                print(f"Saved GIF: {args.save}")
            else:
                raise ValueError("Unsupported extension for --save. Use .mp4 or .gif")
        except Exception as e:
            print(f"Failed to save {args.save} ({e}). Showing interactive window instead...")
            plt.show()
    else:
        plt.show()

if __name__ == "__main__":
    main()
