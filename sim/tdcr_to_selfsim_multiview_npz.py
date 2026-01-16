#!/usr/bin/env python3
"""
Convert a TDCR MuJoCo dataset (multi-view RGB + mask PNG + motor JSON) into the
multi-view NPZ format expected by the modified SelfSimRobot training code.

Expected input layout (from your tdcr_pipeline.py collect):
  <root>/
    motor/                # JSON files per sample, e.g. 000001.json
    rgb/
      <cam_name_0>/       # per-camera folders
        000001.png
        000001_mask.png
        ...
      <cam_name_1>/
        ...

Output NPZ keys:
  images: (N, V, S, S) float32 in {0,1}
  angles: (N, DOF) float32 (optionally normalized)
  rays_o: (V, S*S, 3) float32 (world coords)
  rays_d: (V, S*S, 3) float32 (world dirs)
  near:   (V,) float32
  far:    (V,) float32
  camera_names: (V,) str

Notes:
- This script needs `mujoco` Python package to read camera poses from the XML.
- It assumes cameras are *fixed* (world/static). If your cameras are attached to a moving body,
  you would need to store per-sample camera poses during data collection and adapt this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    """Center-crop a (H,W,...) array to a square using the shorter side."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def _read_mask_png(path: Path, out_size: int, threshold: int = 127) -> np.ndarray:
    """Read a PNG mask and return float32 array (S,S) in {0,1}."""
    im = Image.open(path).convert("L")
    arr = np.asarray(im)  # (H,W) uint8
    arr = _center_crop_square(arr)
    im2 = Image.fromarray(arr)
    im2 = im2.resize((out_size, out_size), resample=Image.NEAREST)
    arr2 = np.asarray(im2)
    mask = (arr2 > threshold).astype(np.float32)
    return mask


def _load_ctrl_from_json(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        d = json.load(f)

    # Try common keys
    for k in ("ctrl", "motor", "motors", "action", "u"):
        if k in d:
            v = d[k]
            if isinstance(v, dict) and "ctrl" in v:
                v = v["ctrl"]
            return np.asarray(v, dtype=np.float32).reshape(-1)

    # Fallback: if dict has a single list-like value
    for v in d.values():
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (int, float)):
            return np.asarray(v, dtype=np.float32).reshape(-1)

    raise ValueError(f"Cannot find motor ctrl in {path}")


def _normalize_ctrl(ctrl: np.ndarray, ctrlrange: np.ndarray, mode: str) -> np.ndarray:
    """
    ctrlrange: (DOF,2) [low, high]
    mode:
      - 'none'
      - 'ctrlrange01'  -> [0,1]
      - 'ctrlrange-11' -> [-1,1]
    """
    if mode == "none":
        return ctrl.astype(np.float32)

    lo = ctrlrange[:, 0].astype(np.float32)
    hi = ctrlrange[:, 1].astype(np.float32)
    denom = np.clip(hi - lo, 1e-8, None)
    x01 = (ctrl - lo) / denom
    if mode == "ctrlrange01":
        return x01.astype(np.float32)
    if mode == "ctrlrange-11":
        return (2.0 * x01 - 1.0).astype(np.float32)

    raise ValueError(f"Unknown normalize_ctrl mode: {mode}")


def _make_rays_for_camera(
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    R_c2w: np.ndarray,
    t_c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create rays in world coordinates for a pinhole camera.

    Conventions:
      - Pixel coords: x right, y down.
      - Camera looks along -Z (OpenGL-style), so we use z = -1 in camera frame.
      - dirs_world = dirs_cam @ R^T  (row-vector convention)

    Returns:
      rays_o: (H*W,3)
      rays_d: (H*W,3)
    """
    # Pixel grid in row-major order (y,x), so flatten matches mask.reshape(-1)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")  # (H,W)

    dirs_cam = np.stack(
        [
            (xs - cx) / fx,
            -(ys - cy) / fy,
            -np.ones_like(xs),
        ],
        axis=-1,
    )  # (H,W,3)

    dirs_world = dirs_cam @ R_c2w.T  # (H,W,3)

    origins = np.broadcast_to(t_c2w.reshape(1, 1, 3), dirs_world.shape).copy()

    rays_o = origins.reshape(-1, 3).astype(np.float32)
    rays_d = dirs_world.reshape(-1, 3).astype(np.float32)
    return rays_o, rays_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root (contains motor/ and rgb/)")
    ap.add_argument("--xml", type=str, required=True, help="MuJoCo XML used for rendering (for camera poses + ctrlrange)")
    ap.add_argument("--out_npz", type=str, required=True, help="Output .npz path")
    ap.add_argument("--rgb_dir", type=str, default=None, help="Override rgb dir (default: <root>/rgb)")
    ap.add_argument("--motor_dir", type=str, default=None, help="Override motor dir (default: <root>/motor)")
    ap.add_argument("--size", type=int, default=100, help="Output mask resolution (square)")
    ap.add_argument("--cameras", type=str, nargs="*", default=None,
                    help="Camera names to include. Default: all subfolders in rgb_dir.")
    ap.add_argument("--mask_suffix", type=str, default="_mask.png", help="Mask filename suffix")
    ap.add_argument("--threshold", type=int, default=127, help="Mask binarization threshold")
    ap.add_argument("--normalize_ctrl", type=str, default="ctrlrange01",
                    choices=["none", "ctrlrange01", "ctrlrange-11"],
                    help="How to normalize motor ctrl values")
    ap.add_argument("--lookat", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="3D point the cameras are roughly looking at (for near/far heuristic)")
    ap.add_argument("--nf_size", type=float, default=0.6,
                    help="near/far half-range: near=dist-nf_size, far=dist+nf_size")
    ap.add_argument("--strict", action="store_true",
                    help="If set, require every sample to have masks for all cameras; else skip incomplete samples")

    args = ap.parse_args()

    root = Path(args.root)
    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else (root / "rgb")
    motor_dir = Path(args.motor_dir) if args.motor_dir else (root / "motor")
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    if not rgb_dir.exists():
        raise FileNotFoundError(f"rgb_dir not found: {rgb_dir}")
    if not motor_dir.exists():
        raise FileNotFoundError(f"motor_dir not found: {motor_dir}")

    # ---- find cameras from folders ----
    cam_folders = [p for p in rgb_dir.iterdir() if p.is_dir()]
    cam_names = sorted([p.name for p in cam_folders])
    if args.cameras and len(args.cameras) > 0:
        requested = args.cameras
        missing = [c for c in requested if (rgb_dir / c).is_dir() is False]
        if missing:
            raise FileNotFoundError(f"Requested camera folders not found under {rgb_dir}: {missing}")
        cam_names = requested

    if len(cam_names) == 0:
        raise RuntimeError(f"No camera folders found under {rgb_dir}")

    V = len(cam_names)
    S = int(args.size)
    print(f"[INFO] cameras ({V}): {cam_names}")
    print(f"[INFO] output mask size: {S}x{S}")

    # ---- load MuJoCo model for camera poses & ctrlrange ----
    try:
        import mujoco
    except Exception as e:
        raise RuntimeError(
            "This script requires `mujoco` python package. "
            "Install mujoco>=2.3.x, then retry."
        ) from e

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # ctrlrange for actuators
    if model.nu <= 0:
        raise RuntimeError("Model has no actuators (nu==0); cannot infer motor ctrlrange.")
    ctrlrange = np.array(model.actuator_ctrlrange, dtype=np.float32)  # (nu,2)

    # ---- precompute rays per camera (fixed cameras) ----
    rays_o_all = []
    rays_d_all = []
    nears = []
    fars = []
    focals = []

    lookat = np.asarray(args.lookat, dtype=np.float32)
    for cam_name in cam_names:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            raise ValueError(f"Camera '{cam_name}' not found in XML cameras.")

        fovy = float(model.cam_fovy[cid])
        if fovy <= 0:
            # fall back to global fovy if per-camera fovy is not set
            try:
                fovy = float(model.vis.global_.fovy)
            except Exception:
                raise ValueError(f"Camera '{cam_name}' has fovy<=0 and no global fovy found.")

        # For square output, treat fovy as vertical fov
        fy = (S / 2.0) / math.tan(math.radians(fovy) / 2.0)
        fx = fy
        cx = (S - 1) / 2.0
        cy = (S - 1) / 2.0
        focals.append(fy)

        # MuJoCo provides camera pose in world coordinates
        t_c2w = np.array(data.cam_xpos[cid], dtype=np.float32)  # (3,)
        R_c2w = np.array(data.cam_xmat[cid], dtype=np.float32).reshape(3, 3)  # (3,3), row-major

        ro, rd = _make_rays_for_camera(S, S, fx, fy, cx, cy, R_c2w, t_c2w)
        rays_o_all.append(ro)
        rays_d_all.append(rd)

        dist = float(np.linalg.norm(t_c2w - lookat))
        near = max(0.01, dist - float(args.nf_size))
        far = dist + float(args.nf_size)
        nears.append(near)
        fars.append(far)

        print(f"[INFO] cam={cam_name:>15s}  fovy={fovy:.1f}  dist~{dist:.3f}  near={near:.3f}  far={far:.3f}")

    rays_o_all = np.stack(rays_o_all, axis=0)  # (V, S*S, 3)
    rays_d_all = np.stack(rays_d_all, axis=0)  # (V, S*S, 3)
    nears = np.asarray(nears, dtype=np.float32)
    fars = np.asarray(fars, dtype=np.float32)
    focals = np.asarray(focals, dtype=np.float32)

    # ---- enumerate samples (motor json files) ----
    motor_files = sorted(motor_dir.glob("*.json"))
    if len(motor_files) == 0:
        raise RuntimeError(f"No motor json files found in {motor_dir}")

    images: List[np.ndarray] = []
    angles: List[np.ndarray] = []
    kept_stems: List[str] = []

    for jpath in motor_files:
        stem = jpath.stem  # e.g. "000001"
        ctrl = _load_ctrl_from_json(jpath)

        if ctrl.shape[0] != model.nu:
            raise ValueError(f"{jpath} ctrl dim {ctrl.shape[0]} != model.nu {model.nu}")

        ctrl_n = _normalize_ctrl(ctrl, ctrlrange, args.normalize_ctrl)

        # Load masks for all cameras
        masks_v = []
        ok = True
        for cam_name in cam_names:
            mpath = rgb_dir / cam_name / f"{stem}{args.mask_suffix}"
            if not mpath.exists():
                ok = False
                if args.strict:
                    raise FileNotFoundError(f"Missing mask: {mpath}")
                break
            masks_v.append(_read_mask_png(mpath, out_size=S, threshold=args.threshold))

        if not ok:
            continue

        images.append(np.stack(masks_v, axis=0))  # (V,S,S)
        angles.append(ctrl_n)
        kept_stems.append(stem)

    if len(images) == 0:
        raise RuntimeError("No complete samples were found (check --strict / camera folders / filenames).")

    images_np = np.stack(images, axis=0).astype(np.float32)  # (N,V,S,S)
    angles_np = np.stack(angles, axis=0).astype(np.float32)  # (N,DOF)

    print(f"[INFO] kept samples: {images_np.shape[0]} / {len(motor_files)}")

    # ---- save npz ----
    np.savez_compressed(
        out_npz,
        images=images_np,
        angles=angles_np,
        rays_o=rays_o_all,
        rays_d=rays_d_all,
        near=nears,
        far=fars,
        focal=focals,  # optional (per-view), kept for debugging/inspection
        camera_names=np.asarray(cam_names),
        stems=np.asarray(kept_stems),
    )
    print(f"[DONE] wrote {out_npz}  (N={images_np.shape[0]}, V={V}, S={S}, DOF={angles_np.shape[1]})")


if __name__ == "__main__":
    main()
'''

python tdcr_to_selfsim_multiview_npz.py \
  --root 2m_no_base \
  --xml tdcr2_no_base.xml \
  --out_npz tdcr_multiview_100.npz \
  --size 100 \
  --normalize_ctrl ctrlrange01 \
  --nf_size 0.6 \
  --strict


'''