from __future__ import annotations

import argparse
import tempfile
import random
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
    _IMAGEIO_OK = True
except Exception:
    _IMAGEIO_OK = False

# Optional PLY readers
_PLYFILE_OK = True
try:
    from plyfile import PlyData  # type: ignore
except Exception:
    _PLYFILE_OK = False

_O3D_OK = True
try:
    import open3d as o3d  # type: ignore
except Exception:
    _O3D_OK = False

# Optional Pillow for advanced GIF compression
_PIL_OK = True
try:
    from PIL import Image  # type: ignore
except Exception:
    _PIL_OK = False


# Match "000001.ply" or "..._000001.ply" etc.
ID_RE = re.compile(r'(?:^|[_\-])(\d{1,})\.ply$', re.IGNORECASE)


@dataclass
class Pair:
    id_str: str
    gt_path: Path
    pred_path: Path


def find_pairs(root: Path, recursive: bool = False) -> List[Pair]:
    """
    Demo format:
      root/
        gt/
          000001.ply
          ...
        pred/
          000001.ply
          ...
    """
    gt_dir = root / "gt"
    pred_dir = root / "pred"
    if not gt_dir.is_dir() or not pred_dir.is_dir():
        raise SystemExit(
            f"Expected folders 'gt' and 'pred' under root.\n"
            f"Got: root={root}\n"
            f"  gt_dir exists? {gt_dir.is_dir()} ({gt_dir})\n"
            f"  pred_dir exists? {pred_dir.is_dir()} ({pred_dir})"
        )

    gt_files = list(gt_dir.rglob("*.ply")) if recursive else list(gt_dir.glob("*.ply"))
    pred_files = list(pred_dir.rglob("*.ply")) if recursive else list(pred_dir.glob("*.ply"))

    gts: Dict[str, Path] = {}
    preds: Dict[str, Path] = {}

    for p in gt_files:
        m = ID_RE.search(p.name.lower())
        if not m:
            continue
        id_str = m.group(1).zfill(6)
        gts[id_str] = p

    for p in pred_files:
        m = ID_RE.search(p.name.lower())
        if not m:
            continue
        id_str = m.group(1).zfill(6)
        preds[id_str] = p

    shared = sorted(set(gts).intersection(preds))
    return [Pair(id_str=s, gt_path=gts[s], pred_path=preds[s]) for s in shared]


def load_ply_xyz(path: Path, max_points: Optional[int] = None) -> np.ndarray:
    pts: Optional[np.ndarray] = None
    if _PLYFILE_OK:
        try:
            ply = PlyData.read(str(path))
            v = ply["vertex"]
            pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
        except Exception:
            pts = None
    if pts is None and _O3D_OK:
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(pcd.points, dtype=np.float32)
        except Exception:
            pts = None
    if pts is None:
        raise RuntimeError(f"Failed to read PLY: {path}. Install 'plyfile' or 'open3d'.")

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32)


def center_pair(gt: np.ndarray, pred: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    保留原来的 center 选项（不做 ICP / mirror 等对齐），只做可选的平移居中。
    """
    if mode == "gt":
        c = gt.mean(axis=0, keepdims=True)
        return gt - c, pred - c
    elif mode == "both":
        return gt - gt.mean(axis=0, keepdims=True), pred - pred.mean(axis=0, keepdims=True)
    else:
        return gt, pred


def chamfer_l2_sq(a: np.ndarray, b: np.ndarray) -> float:
    tb = cKDTree(b)
    da, _ = tb.query(a, k=1)
    ta = cKDTree(a)
    db, _ = ta.query(b, k=1)
    return float((np.mean(da**2) + np.mean(db**2)) * 0.5)


def set_axes_dark(ax):
    ax.set_facecolor("black")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0, 0, 0, 1.0))
        axis._axinfo["grid"]["color"] = (0.2, 0.2, 0.2, 0.3)
        axis.set_ticklabels([])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def set_axes_equal(ax, limits):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = limits
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)


def compute_global_limits(processed):
    xs, ys, zs = [], [], []
    for item in processed:
        pts = np.vstack([item["gt"], item["pred"]])
        xs.extend(pts[:, 0]); ys.extend(pts[:, 1]); zs.extend(pts[:, 2])
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    zmin, zmax = np.min(zs), np.max(zs)
    cx = 0.5 * (xmin + xmax); cy = 0.5 * (ymin + ymax); cz = 0.5 * (zmin + zmax)
    rx = 0.5 * (xmax - xmin); ry = 0.5 * (ymax - ymin); rz = 0.5 * (zmax - zmin)
    r = max(rx, ry, rz) * 1.05 + 1e-6
    return ((cx - r, cx + r), (cy - r, cy + r), (cz - r, cz + r))


def render_grid_png(processed, out_path: Path, pt_size: float, elev: float, azim: float, dpi: int, title: str):
    fig = plt.figure(figsize=(14, 12), facecolor="black", dpi=dpi)
    fig.suptitle(title, color="white", fontsize=12)
    limits = compute_global_limits(processed)
    for i, item in enumerate(processed):
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        set_axes_dark(ax); set_axes_equal(ax, limits); ax.view_init(elev=elev, azim=azim)
        gt = item["gt"]; pred = item["pred"]
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=pt_size, c="#FFFFFF", alpha=0.95, linewidths=0)
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=pt_size, c="#00E5FF", alpha=0.95, linewidths=0)
        subtitle = f"{item['id']} | CD^2={item['cd2']:.2e}"
        ax.text2D(0.02, 0.98, subtitle, transform=ax.transAxes, ha="left", va="top", color="#DDDDDD", fontsize=8)
    lines = [plt.Line2D([0], [0], marker='o', color='w', label='GT', markerfacecolor="#FFFFFF", markersize=6),
             plt.Line2D([0], [0], marker='o', color='w', label='Pred', markerfacecolor="#00E5FF", markersize=6)]
    fig.legend(handles=lines, loc="lower center", ncol=2, frameon=False, labelcolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.04, wspace=0.02, hspace=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)


def render_grid_gif(processed, gif_path: Path, pt_size: float, elev: float, azim0: float, orbit: float,
                    frames: int, fps: int, dpi: int, title: str):
    if not _IMAGEIO_OK:
        raise RuntimeError("imageio is required for GIF export. Install with: pip install imageio")
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        frame_paths = []
        for f in range(frames):
            azim = azim0 + (orbit * f / frames)
            png_path = tmp_dir / f"frame_{f:04d}.png"
            render_grid_png(processed, png_path, pt_size, elev, azim, dpi, title)
            frame_paths.append(png_path)
        images = [imageio.imread(p) for p in frame_paths]
        duration = max(1, int(1000 / max(1, fps)))
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(gif_path), images, duration=duration, loop=0)


def _save_gif_pillow(frames: List["Image.Image"], path: Path, fps: int):
    duration = max(1, int(1000 / max(1, fps)))
    frames_p = [f.convert("P", palette=Image.ADAPTIVE, colors=256) if f.mode != "P" else f for f in frames]
    frames_p[0].save(
        str(path),
        save_all=True,
        append_images=frames_p[1:],
        loop=0,
        duration=duration,
        optimize=True,
        disposal=2,
    )


def render_grid_gif_small(
    processed,
    gif_path: Path,
    pt_size: float,
    elev: float,
    azim0: float,
    orbit: float,
    frames: int,
    fps: int,
    dpi: int,
    title: str,
    target_mb: float = 5.0,
    scale: float = 0.70,
    palette: int = 128,
    dither: str = "floyd",
    tries: int = 3,
    drop_step_start: int = 1,
):
    if not _PIL_OK:
        print("[ninegrid] NOTE: Pillow not available; using fallback GIF without palette quantization.")
        render_grid_gif(processed, gif_path, pt_size, elev, azim0, orbit, frames, fps, dpi, title)
        return

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        src_images: List[Image.Image] = []
        for f in range(frames):
            azim = azim0 + (orbit * f / max(1, frames))
            png_path = tmp_dir / f"frame_{f:04d}.png"
            render_grid_png(processed, png_path, pt_size, elev, azim, dpi, title)
            img = Image.open(png_path).convert("RGB")
            src_images.append(img)

        cur_scale = float(scale)
        cur_palette = int(max(2, min(256, palette)))
        cur_fps = int(max(1, fps))
        drop_step = int(max(1, drop_step_start))

        for t in range(max(1, tries)):
            frames_this_try: List[Image.Image] = []
            for idx in range(0, len(src_images), drop_step):
                img = src_images[idx]
                if cur_scale != 1.0:
                    w = max(1, int(img.width * cur_scale))
                    h = max(1, int(img.height * cur_scale))
                    img = img.resize((w, h), resample=Image.LANCZOS)
                dither_flag = Image.FLOYDSTEINBERG if dither == "floyd" else Image.NONE
                img_q = img.quantize(colors=max(2, min(256, cur_palette)), method=Image.MEDIANCUT, dither=dither_flag)
                frames_this_try.append(img_q)

            gif_path.parent.mkdir(parents=True, exist_ok=True)
            _save_gif_pillow(frames_this_try, gif_path, cur_fps)
            size_mb = os.path.getsize(gif_path) / (1024 * 1024)
            print(f"[ninegrid] small GIF try {t+1}/{tries}: size={size_mb:.2f} MB  (scale={cur_scale:.2f}, palette={cur_palette}, fps={cur_fps}, step={drop_step})")

            if size_mb <= target_mb or t == tries - 1:
                print(f"[ninegrid] small GIF saved: {gif_path.resolve()}  (final size={size_mb:.2f} MB)")
                break

            cur_scale *= 0.85
            cur_palette = max(16, cur_palette // 2)
            if cur_fps > 6:
                cur_fps = max(6, int(cur_fps * 0.9))
            drop_step += 1


def process_pairs(pairs: List[Pair], take: int, sample_mode: str, center: str,
                  max_points: Optional[int]):
    if sample_mode == "random":
        random.seed(42); random.shuffle(pairs)
    picked = pairs[:take]
    processed = []
    for p in picked:
        gt = load_ply_xyz(p.gt_path, max_points=max_points)
        pred = load_ply_xyz(p.pred_path, max_points=max_points)
        gt, pred = center_pair(gt, pred, center)
        cd2 = chamfer_l2_sq(gt, pred)
        processed.append({"id": p.id_str, "gt": gt, "pred": pred, "cd2": cd2})
    return processed


def parse_args():
    ap = argparse.ArgumentParser(description="Create a 3x3 collage (and optional rotating GIF) of GT vs Pred point clouds")
    ap.add_argument("--root", type=str, required=True,
                    help="Demo output root folder containing subfolders: root/gt/*.ply and root/pred/*.ply")
    ap.add_argument("--recursive", action="store_true", help="Search recursively under gt/ and pred/")
    ap.add_argument("--take", type=int, default=9, help="How many pairs to render (max 9 fits the grid)")
    ap.add_argument("--sample-mode", choices=["first", "random"], default="first", help="Pick first N or random N")
    ap.add_argument("--max-points", type=int, default=8192, help="Downsample each cloud to at most N points")

    # 只保留可选居中（不做任何 ICP/mirror 对齐）
    ap.add_argument("--center", choices=["gt", "both", "none"], default="none", help="Centering mode")
    ap.add_argument("--pt-size", type=float, default=1.8, help="Marker size")

    ap.add_argument("--out", type=str, default="ninegrid.png", help="PNG output path")
    ap.add_argument("--gif", type=str, default="", help="GIF output path (optional)")
    ap.add_argument("--dpi", type=int, default=150, help="Render DPI")

    ap.add_argument("--frames", type=int, default=90, help="GIF: number of frames")
    ap.add_argument("--fps", type=int, default=18, help="GIF: frames per second")
    ap.add_argument("--elev", type=float, default=20.0, help="Camera elevation (deg)")
    ap.add_argument("--azim", type=float, default=45.0, help="Camera starting azimuth (deg)")
    ap.add_argument("--orbit", type=float, default=360.0, help="GIF: azimuth sweep (deg)")

    # --- small/compressed GIF options ---
    ap.add_argument("--gif-small", type=str, default="", help="Optional: output a compressed small GIF (tries to be <= target MB)")
    ap.add_argument("--gif-small-target-mb", type=float, default=5.0, help="Target size for the small GIF in MB")
    ap.add_argument("--gif-small-dpi", type=int, default=100, help="DPI used to render frames for the small GIF (smaller => fewer pixels)")
    ap.add_argument("--gif-small-scale", type=float, default=0.70, help="Extra downscale factor after rendering (e.g., 0.70)")
    ap.add_argument("--gif-small-fps", type=int, default=12, help="FPS for the small GIF")
    ap.add_argument("--gif-small-frames", type=int, default=60, help="Total frames for the small GIF (<= --frames recommended)")
    ap.add_argument("--gif-small-palette", type=int, default=128, help="Palette size for quantization (max 256)")
    ap.add_argument("--gif-small-tries", type=int, default=3, help="If above target, number of progressive compression attempts")
    ap.add_argument("--gif-small-dither", choices=["floyd", "none"], default="floyd", help="Dither mode during quantization")

    ap.add_argument("--debug", action="store_true", help="Print debug info if no pairs found")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    assert root.exists(), f"Root not found: {root}"

    pairs = find_pairs(root, recursive=args.recursive)
    if not pairs:
        if args.debug:
            gt_dir = root / "gt"
            pred_dir = root / "pred"
            gt_plys = list(gt_dir.rglob("*.ply")) if gt_dir.is_dir() else []
            pred_plys = list(pred_dir.rglob("*.ply")) if pred_dir.is_dir() else []
            print(f"[debug] root={root}")
            print(f"[debug] gt_dir={gt_dir} exists={gt_dir.is_dir()}  ply_count={len(gt_plys)}")
            print(f"[debug] pred_dir={pred_dir} exists={pred_dir.is_dir()}  ply_count={len(pred_plys)}")
            if gt_plys:
                print("[debug] example gt files:", [p.name for p in gt_plys[:10]])
            if pred_plys:
                print("[debug] example pred files:", [p.name for p in pred_plys[:10]])
            print("[debug] Expect filenames like 000123.ply under both gt/ and pred/.")
        raise SystemExit(f"No pairs found in {root} (need gt/XXXXXX.ply + pred/XXXXXX.ply)")

    take = min(args.take, 9)
    processed = process_pairs(pairs, take, args.sample_mode, args.center, args.max_points)

    title = f"Nine-Grid GT vs Pred  •  root={root.name}  •  center={args.center}"
    out_png = Path(args.out)
    render_grid_png(processed, out_png, args.pt_size, args.elev, args.azim, args.dpi, title)
    print(f"[ninegrid] PNG saved: {out_png.resolve()}")

    if args.gif:
        out_gif = Path(args.gif)
        render_grid_gif(processed, out_gif, args.pt_size, args.elev, args.azim, args.orbit, args.frames, args.fps, args.dpi, title)
        size_mb = os.path.getsize(out_gif) / (1024 * 1024) if out_gif.exists() else -1
        print(f"[ninegrid] GIF saved: {out_gif.resolve()} (size={size_mb:.2f} MB)")

    if args.gif_small:
        out_small = Path(args.gif_small)
        frames_small = max(1, min(args.frames, args.gif_small_frames))
        render_grid_gif_small(
            processed,
            out_small,
            args.pt_size,
            args.elev,
            args.azim,
            args.orbit,
            frames_small,
            args.gif_small_fps,
            args.gif_small_dpi,
            title,
            target_mb=args.gif_small_target_mb,
            scale=args.gif_small_scale,
            palette=args.gif_small_palette,
            dither=args.gif_small_dither,
            tries=args.gif_small_tries,
        )
        size_mb = os.path.getsize(out_small) / (1024 * 1024) if out_small.exists() else -1
        print(f"[ninegrid] Small GIF saved: {out_small.resolve()} (size={size_mb:.2f} MB, target={args.gif_small_target_mb:.2f} MB)")


if __name__ == "__main__":
    main()


"""
python pc_ninegrid_demo.py \
  --root demo_out/sim5_with_base_mlp \
  --out ninegrid.png \
  --gif ninegrid.gif \
  --take 9 --sample-mode first \
  --center none \
  --pt-size 1.8 --max-points 4096 \
  --frames 90 --fps 18 --elev 20 --orbit 360
"""
