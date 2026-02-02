#!/usr/bin/env python3
"""
tdcr_to_selfsim_multiview_npz.py

Convert a TDCR MuJoCo dataset (multi-view mask PNG + motor JSON + (optional) pointcloud PLY)
into the multi-view NPZ format expected by the modified SelfSimRobot training code.

This v2 adds:
  - Auto near/far estimation from per-sample point clouds + camera poses (recommended for 3m/5m)
  - Robust percentile-based bounds to avoid outliers (e.g., floor/base points)

Expected input layout (from your tdcr_pipeline.py collect):
  <root>/
    motor/                # JSON files per sample, e.g. 000001.json
    rgb/
      <cam_name_0>/       # per-camera folders
        000001_mask.png
        ...
    pointcloud/           # OPTIONAL but recommended for auto near/far
      000001.ply
      ...

Output NPZ keys:
  images: (N, V, S, S) float32 in {0,1}
  angles: (N, DOF) float32 (optionally normalized)
  rays_o: (V, S*S, 3) float32 (world coords)
  rays_d: (V, S*S, 3) float32 (world dirs, not normalized; z-component in cam frame is -1)
  near:   (V,) float32      (per-camera near plane)
  far:    (V,) float32      (per-camera far plane)
  camera_names: (V,) str
  stems: (N,) str
  meta: json string (optional; for debugging)

Notes:
- Requires `mujoco` python package to read camera poses from the XML.
- Assumes cameras are *fixed* (world/static).
- If you used your collect pipeline to export point clouds by backprojecting MuJoCo depth,
  the point coordinates are in world frame and consistent with data.cam_xpos / data.cam_xmat.

Near/Far conventions:
- The training code samples points as: p(t) = o + t * d.
- Our rays are built with camera-forward = -Z (OpenGL-like), so in camera frame d_z = -1.
  That makes the ray parameter t equal to the *depth value* used in backprojection.
- Therefore, computing near/far from point clouds should use "depth along camera forward axis":
    depth = -z_cam = -((p_world - cam_pos) @ R_c2w)[2]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image


# -----------------------------
# Utilities: masks
# -----------------------------
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


# -----------------------------
# Utilities: motors
# -----------------------------
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


# -----------------------------
# Utilities: rays
# -----------------------------
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

    IMPORTANT:
      We intentionally DO NOT normalize directions.
      Because d_z = -1 in camera frame, the ray parameter t corresponds to the same "depth"
      value used by MuJoCo depth backprojection (see collect.py).
    """
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


# -----------------------------
# Utilities: point cloud PLY reader (xyz only)
# -----------------------------
def _load_ply_xyz(path: Path, max_points: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """
    Load point cloud xyz from a PLY file.

    - Tries open3d first (if installed).
    - Falls back to a minimal PLY parser supporting:
        * binary_little_endian
        * ascii
      with vertex properties including x,y,z (float32/float64).
    """
    # 1) open3d fast-path
    try:
        import open3d as o3d  # type: ignore
        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
    except Exception:
        xyz = _load_ply_xyz_fallback(path)

    if xyz.size == 0:
        return xyz.astype(np.float32)

    # remove non-finite
    m = np.isfinite(xyz).all(axis=1)
    xyz = xyz[m]
    if xyz.size == 0:
        return xyz.astype(np.float32)

    # downsample for speed (deterministic)
    if (max_points is not None) and (max_points > 0) and (xyz.shape[0] > max_points):
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], size=int(max_points), replace=False)
        xyz = xyz[idx]

    return xyz.astype(np.float32)


def _load_ply_xyz_fallback(path: Path) -> np.ndarray:
    import re
    import struct

    with open(path, "rb") as f:
        header_lines: List[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (unexpected EOF): {path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        header = b"".join(header_lines).decode("ascii", errors="ignore")
        # format
        m = re.search(r"format\s+(\w+)\s+1\.0", header)
        if not m:
            raise ValueError(f"Cannot parse PLY format: {path}")
        fmt = m.group(1).strip()

        # vertex count
        m = re.search(r"element\s+vertex\s+(\d+)", header)
        if not m:
            raise ValueError(f"Cannot parse vertex count: {path}")
        n_vert = int(m.group(1))

        # parse properties of vertex element
        # We assume standard layout where vertex properties are listed immediately after "element vertex"
        props: List[Tuple[str, str]] = []  # (type, name)
        in_vertex = False
        for line in header.splitlines():
            if line.startswith("element vertex"):
                in_vertex = True
                continue
            if line.startswith("element ") and in_vertex:
                # next element begins
                break
            if in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    ptype = parts[1].strip()
                    pname = parts[2].strip()
                    props.append((ptype, pname))

        if len(props) == 0:
            raise ValueError(f"No vertex properties found in PLY header: {path}")

        # map ply types -> numpy dtypes (binary)
        ply2np = {
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
            "uchar": "u1",
            "uint8": "u1",
            "char": "i1",
            "int8": "i1",
            "ushort": "<u2",
            "uint16": "<u2",
            "short": "<i2",
            "int16": "<i2",
            "uint": "<u4",
            "uint32": "<u4",
            "int": "<i4",
            "int32": "<i4",
        }

        # figure x,y,z indices
        names = [p[1] for p in props]
        try:
            ix = names.index("x")
            iy = names.index("y")
            iz = names.index("z")
        except ValueError:
            raise ValueError(f"PLY missing x/y/z properties: {path}; props={names[:10]}...")

        if fmt == "ascii":
            # read as text, take first 3 columns corresponding to x,y,z positions in property order
            # We load all numeric columns then slice
            # (This is slower but works for small PLYs.)
            txt = f.read().decode("ascii", errors="ignore").strip().splitlines()
            if len(txt) < n_vert:
                raise ValueError(f"ASCII PLY has fewer lines than vertices: {path}")
            arr = np.loadtxt(txt[:n_vert], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            xyz = np.stack([arr[:, ix], arr[:, iy], arr[:, iz]], axis=1)
            return xyz.astype(np.float32)

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format (only ascii/binary_little_endian): {fmt} in {path}")

        # binary little endian
        dtype_fields = []
        for ptype, pname in props:
            if ptype.startswith("list"):
                raise ValueError(f"PLY list properties not supported (got '{ptype} {pname}') in {path}")
            if ptype not in ply2np:
                raise ValueError(f"Unsupported PLY property type '{ptype}' in {path}")
            dtype_fields.append((pname, ply2np[ptype]))
        dt = np.dtype(dtype_fields)

        data = np.fromfile(f, dtype=dt, count=n_vert)
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
        return xyz


# -----------------------------
# Auto near/far from point clouds
# -----------------------------
def _compute_near_far_from_pointclouds(
    stems: List[str],
    pc_dir: Path,
    pc_ext: str,
    cam_R_c2w: np.ndarray,   # (V,3,3)
    cam_t_c2w: np.ndarray,   # (V,3)
    *,
    max_points_per_cloud: int,
    seed: int,
    depth_min: float,
    per_cloud_qmin: float,
    per_cloud_qmax: float,
    far_quantile: float,
    margin: float,
    near_floor: float,
    far_max: Optional[float],
    strict: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Estimate per-camera near/far planes from world-frame point clouds.

    For each stem and each camera:
      1) load xyz
      2) compute depth in that camera: depth = -z_cam
      3) take robust bounds within the cloud: [p(qmin), p(qmax)]
    Then aggregate over stems:
      near = min(dmin) - margin
      far  = percentile(dmax, far_quantile) + margin  (robust to outliers)
    """
    V = int(cam_R_c2w.shape[0])
    dmin_lists: List[List[float]] = [[] for _ in range(V)]
    dmax_lists: List[List[float]] = [[] for _ in range(V)]

    # deterministic rng for subsampling within each cloud
    rng_global = np.random.default_rng(seed)

    # Optional tqdm
    try:
        from tqdm import tqdm  # type: ignore
        it = tqdm(stems, desc="[auto_nf] scanning pointclouds", ncols=120)
    except Exception:
        it = stems

    missing = 0
    used = 0

    for stem in it:
        p = pc_dir / f"{stem}{pc_ext}"
        if not p.exists():
            missing += 1
            if strict:
                raise FileNotFoundError(f"Missing point cloud: {p}")
            continue

        # Make per-stem deterministic seed for point subsampling
        stem_seed = int(rng_global.integers(0, 2**31 - 1))
        xyz = _load_ply_xyz(p, max_points=max_points_per_cloud, seed=stem_seed)
        if xyz.size == 0:
            continue

        used += 1
        for vid in range(V):
            R = cam_R_c2w[vid].astype(np.float32)
            t = cam_t_c2w[vid].astype(np.float32)

            # cam coords (row-vector): cam = (world - t) @ R
            cam = (xyz - t[None, :]) @ R
            depth = (-cam[:, 2]).astype(np.float32)

            # keep points in front of camera
            depth = depth[depth > float(depth_min)]
            if depth.size == 0:
                continue

            # robust within-cloud bounds
            dmin = float(np.percentile(depth, per_cloud_qmin))
            dmax = float(np.percentile(depth, per_cloud_qmax))
            if not np.isfinite(dmin) or not np.isfinite(dmax):
                continue
            if dmax <= dmin:
                continue
            dmin_lists[vid].append(dmin)
            dmax_lists[vid].append(dmax)

    nears = np.zeros((V,), dtype=np.float32)
    fars = np.zeros((V,), dtype=np.float32)

    stats: Dict[str, object] = {
        "pc_dir": str(pc_dir),
        "pc_ext": str(pc_ext),
        "stems_total": int(len(stems)),
        "pc_missing": int(missing),
        "pc_used": int(used),
        "depth_min": float(depth_min),
        "per_cloud_qmin": float(per_cloud_qmin),
        "per_cloud_qmax": float(per_cloud_qmax),
        "far_quantile": float(far_quantile),
        "margin": float(margin),
        "near_floor": float(near_floor),
        "far_max": (None if far_max is None else float(far_max)),
    }

    for vid in range(V):
        if len(dmin_lists[vid]) == 0 or len(dmax_lists[vid]) == 0:
            nears[vid] = float(near_floor)
            fars[vid] = float(near_floor + 1.0)
            continue

        dmin_arr = np.asarray(dmin_lists[vid], dtype=np.float32)
        dmax_arr = np.asarray(dmax_lists[vid], dtype=np.float32)

        near_v = float(np.min(dmin_arr) - margin)
        far_v = float(np.percentile(dmax_arr, far_quantile) + margin)

        near_v = max(float(near_floor), near_v)
        if far_max is not None:
            far_v = min(float(far_max), far_v)
        if far_v <= near_v + 1e-6:
            far_v = near_v + 1e-3

        nears[vid] = near_v
        fars[vid] = far_v

        # per-view debug stats
        stats[f"view{vid}_dmin_min/med/max"] = [float(np.min(dmin_arr)), float(np.median(dmin_arr)), float(np.max(dmin_arr))]
        stats[f"view{vid}_dmax_min/med/max"] = [float(np.min(dmax_arr)), float(np.median(dmax_arr)), float(np.max(dmax_arr))]
        stats[f"view{vid}_counts"] = int(len(dmin_arr))

    return nears, fars, stats


def main():
    ap = argparse.ArgumentParser()

    # IO
    ap.add_argument("--root", type=str, required=True, help="Dataset root (contains motor/ and rgb/)")
    ap.add_argument("--xml", type=str, required=True, help="MuJoCo XML used for rendering (camera poses + ctrlrange)")
    ap.add_argument("--out_npz", type=str, required=True, help="Output .npz path")
    ap.add_argument("--rgb_dir", type=str, default=None, help="Override rgb dir (default: <root>/rgb)")
    ap.add_argument("--motor_dir", type=str, default=None, help="Override motor dir (default: <root>/motor)")
    ap.add_argument("--pc_dir", type=str, default=None,
                    help="Point cloud dir for auto near/far (default: <root>/pointcloud)")
    ap.add_argument("--pc_ext", type=str, default=".ply", help="Point cloud extension (default: .ply)")

    # images
    ap.add_argument("--size", type=int, default=100, help="Output mask resolution (square)")
    ap.add_argument("--cameras", type=str, nargs="*", default=None,
                    help="Camera names to include. Default: all subfolders in rgb_dir.")
    ap.add_argument("--mask_suffix", type=str, default="_mask.png", help="Mask filename suffix")
    ap.add_argument("--threshold", type=int, default=127, help="Mask binarization threshold")
    ap.add_argument("--strict", action="store_true",
                    help="If set, require every sample to have masks for all cameras; else skip incomplete samples")

    # motors
    ap.add_argument("--normalize_ctrl", type=str, default="ctrlrange01",
                    choices=["none", "ctrlrange01", "ctrlrange-11"],
                    help="How to normalize motor ctrl values")
    ap.add_argument("--clip_ctrl", action="store_true",
                    help="Clip normalized ctrl to [0,1] or [-1,1] according to normalize mode")

    # near/far (legacy heuristic)
    ap.add_argument("--lookat", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="3D point cameras roughly look at (legacy near/far heuristic)")
    ap.add_argument("--nf_size", type=float, default=0.6,
                    help="Legacy near/far half-range: near=dist-nf_size, far=dist+nf_size")

    ap.add_argument("--near_override", type=float, default=None,
                    help="Override near plane for ALL cameras (takes precedence at the end)")
    ap.add_argument("--far_override", type=float, default=None,
                    help="Override far plane for ALL cameras (takes precedence at the end)")
    ap.add_argument("--far_max", type=float, default=None,
                    help="Clamp far to at most this value AFTER computing it")

    # auto near/far from point clouds
    ap.add_argument("--auto_near_far", action="store_true",
                    help="Compute near/far from point clouds + camera poses (recommended for 3m/5m)")
    ap.add_argument("--nf_max_points_per_cloud", type=int, default=200000,
                    help="Max points loaded per point cloud for near/far estimation (speed/memory)")
    ap.add_argument("--nf_seed", type=int, default=42, help="Seed for point cloud subsampling")
    ap.add_argument("--nf_depth_min", type=float, default=1e-4,
                    help="Minimum valid depth in front of camera when estimating near/far")
    ap.add_argument("--nf_per_cloud_qmin", type=float, default=1.0,
                    help="Per-cloud lower percentile for depth bounds (robust to outliers)")
    ap.add_argument("--nf_per_cloud_qmax", type=float, default=99.0,
                    help="Per-cloud upper percentile for depth bounds (robust to outliers)")
    ap.add_argument("--nf_far_quantile", type=float, default=99.5,
                    help="Across-cloud percentile for FAR (robust to outliers like floor)")
    ap.add_argument("--nf_margin", type=float, default=0.02,
                    help="Safety margin added to both sides: near-=margin, far+=margin")
    ap.add_argument("--nf_near_floor", type=float, default=0.01,
                    help="Minimum allowed near value")
    ap.add_argument("--nf_nsample", type=int, default=0,
                    help="If >0, randomly sample this many stems to estimate near/far (0=use all)")

    args = ap.parse_args()

    root = Path(args.root)
    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else (root / "rgb")
    motor_dir = Path(args.motor_dir) if args.motor_dir else (root / "motor")
    pc_dir = Path(args.pc_dir) if args.pc_dir else (root / "pointcloud")
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
        missing = [c for c in requested if not (rgb_dir / c).is_dir()]
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
            "This script requires `mujoco` python package. Install mujoco>=2.3.x, then retry."
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
    cam_pos_all = []
    cam_R_all = []
    cam_fovy_all = []

    lookat = np.asarray(args.lookat, dtype=np.float32)

    meta: Dict[str, object] = {
        "root": str(root),
        "rgb_dir": str(rgb_dir),
        "motor_dir": str(motor_dir),
        "pc_dir": str(pc_dir),
        "pc_ext": str(args.pc_ext),
        "xml": str(args.xml),
        "size": int(S),
        "mask_suffix": str(args.mask_suffix),
        "threshold": int(args.threshold),
        "normalize_ctrl": str(args.normalize_ctrl),
        "lookat": [float(x) for x in args.lookat],
        "nf_size": float(args.nf_size),
        "near_override": (None if args.near_override is None else float(args.near_override)),
        "far_override": (None if args.far_override is None else float(args.far_override)),
        "far_max": (None if args.far_max is None else float(args.far_max)),
        "clip_ctrl": bool(args.clip_ctrl),
        "cameras": list(cam_names),
        "auto_near_far": bool(args.auto_near_far),
        "auto_nf_cfg": {
            "nf_max_points_per_cloud": int(args.nf_max_points_per_cloud),
            "nf_seed": int(args.nf_seed),
            "nf_depth_min": float(args.nf_depth_min),
            "nf_per_cloud_qmin": float(args.nf_per_cloud_qmin),
            "nf_per_cloud_qmax": float(args.nf_per_cloud_qmax),
            "nf_far_quantile": float(args.nf_far_quantile),
            "nf_margin": float(args.nf_margin),
            "nf_near_floor": float(args.nf_near_floor),
            "nf_nsample": int(args.nf_nsample),
        },
    }

    for cam_name in cam_names:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            raise ValueError(f"Camera '{cam_name}' not found in XML cameras.")

        fovy = float(model.cam_fovy[cid])
        if fovy <= 0:
            try:
                fovy = float(model.vis.global_.fovy)
            except Exception:
                raise ValueError(f"Camera '{cam_name}' has fovy<=0 and no global fovy found.")
        cam_fovy_all.append(float(fovy))

        fy = (S / 2.0) / math.tan(math.radians(fovy) / 2.0)
        fx = fy
        cx = (S - 1) / 2.0
        cy = (S - 1) / 2.0
        focals.append(fy)

        t_c2w = np.array(data.cam_xpos[cid], dtype=np.float32)  # (3,)
        R_c2w = np.array(data.cam_xmat[cid], dtype=np.float32).reshape(3, 3)  # (3,3)

        cam_pos_all.append(t_c2w)
        cam_R_all.append(R_c2w)

        ro, rd = _make_rays_for_camera(S, S, fx, fy, cx, cy, R_c2w, t_c2w)
        rays_o_all.append(ro)
        rays_d_all.append(rd)

        # Legacy heuristic: distance to lookat +/- nf_size
        dist = float(np.linalg.norm(t_c2w - lookat))
        near = max(float(args.nf_near_floor), dist - float(args.nf_size))
        far = dist + float(args.nf_size)
        nears.append(near)
        fars.append(far)

        print(f"[INFO] cam={cam_name:>15s}  fovy={fovy:.1f}  dist~{dist:.3f}  heuristic_near={near:.3f}  heuristic_far={far:.3f}")

    rays_o_all = np.stack(rays_o_all, axis=0).astype(np.float32)  # (V, S*S, 3)
    rays_d_all = np.stack(rays_d_all, axis=0).astype(np.float32)  # (V, S*S, 3)
    nears = np.asarray(nears, dtype=np.float32)
    fars = np.asarray(fars, dtype=np.float32)
    focals = np.asarray(focals, dtype=np.float32)
    cam_pos_all = np.stack(cam_pos_all, axis=0).astype(np.float32)  # (V,3)
    cam_R_all = np.stack(cam_R_all, axis=0).astype(np.float32)      # (V,3,3)
    cam_fovy_all = np.asarray(cam_fovy_all, dtype=np.float32)

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
        if args.clip_ctrl:
            if args.normalize_ctrl == "ctrlrange01":
                ctrl_n = np.clip(ctrl_n, 0.0, 1.0)
            elif args.normalize_ctrl == "ctrlrange-11":
                ctrl_n = np.clip(ctrl_n, -1.0, 1.0)

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

    # ---- auto near/far from point clouds (optional) ----
    auto_nf_stats = None
    if args.auto_near_far:
        if not pc_dir.exists():
            raise FileNotFoundError(
                f"--auto_near_far requires point clouds, but pc_dir not found: {pc_dir}\n"
                f"Either pass --pc_dir explicitly, or disable --auto_near_far."
            )

        stems_nf = kept_stems
        if args.nf_nsample and args.nf_nsample > 0 and args.nf_nsample < len(stems_nf):
            rng = np.random.default_rng(int(args.nf_seed))
            sel = rng.choice(len(stems_nf), size=int(args.nf_nsample), replace=False)
            stems_nf = [stems_nf[i] for i in sel]
            print(f"[INFO] auto_nf: using random subset stems: {len(stems_nf)} / {len(kept_stems)}")
        else:
            print(f"[INFO] auto_nf: using all stems: {len(stems_nf)}")

        nears_pc, fars_pc, auto_nf_stats = _compute_near_far_from_pointclouds(
            stems_nf,
            pc_dir=pc_dir,
            pc_ext=str(args.pc_ext),
            cam_R_c2w=cam_R_all,
            cam_t_c2w=cam_pos_all,
            max_points_per_cloud=int(args.nf_max_points_per_cloud),
            seed=int(args.nf_seed),
            depth_min=float(args.nf_depth_min),
            per_cloud_qmin=float(args.nf_per_cloud_qmin),
            per_cloud_qmax=float(args.nf_per_cloud_qmax),
            far_quantile=float(args.nf_far_quantile),
            margin=float(args.nf_margin),
            near_floor=float(args.nf_near_floor),
            far_max=(None if args.far_max is None else float(args.far_max)),
            strict=bool(args.strict),
        )

        nears = nears_pc
        fars = fars_pc

        print("[INFO] auto_nf computed near/far (before overrides):")
        for vid, cam_name in enumerate(cam_names):
            print(f"  cam={cam_name:>15s}  near={float(nears[vid]):.4f}  far={float(fars[vid]):.4f}  "
                  f"(count={auto_nf_stats.get(f'view{vid}_counts', 0) if auto_nf_stats else 0})")

    # ---- apply user overrides (always take precedence) ----
    if args.near_override is not None:
        nears = np.asarray([float(args.near_override)] * V, dtype=np.float32)
    if args.far_override is not None:
        fars = np.asarray([float(args.far_override)] * V, dtype=np.float32)
    if args.far_max is not None:
        fars = np.minimum(fars, float(args.far_max)).astype(np.float32)

    # Safety: ensure far>near
    for vid in range(V):
        if not (float(fars[vid]) > float(nears[vid]) + 1e-6):
            fars[vid] = nears[vid] + 1e-3

    # Quick stats
    img_mean = float(images_np.mean())
    ang_min = float(np.min(angles_np))
    ang_max = float(np.max(angles_np))
    ang_std = float(np.std(angles_np))
    print(f"[STATS] images mean={img_mean:.6f}  angles min/max/std={ang_min:.6f}/{ang_max:.6f}/{ang_std:.6f}")
    print(f"[STATS] near range: min={float(np.min(nears)):.4f} max={float(np.max(nears)):.4f}")
    print(f"[STATS] far  range: min={float(np.min(fars)):.4f} max={float(np.max(fars)):.4f}")

    if auto_nf_stats is not None:
        meta["auto_nf_stats"] = auto_nf_stats

    # ---- save npz ----
    np.savez_compressed(
        out_npz,
        images=images_np,
        angles=angles_np,
        rays_o=rays_o_all,
        rays_d=rays_d_all,
        near=nears.astype(np.float32),
        far=fars.astype(np.float32),
        focal=focals,  # optional (per-view), kept for debugging/inspection
        camera_names=np.asarray(cam_names),
        stems=np.asarray(kept_stems),
        # extra camera pose meta (optional)
        cam_pos=cam_pos_all,     # (V,3)
        cam_R=cam_R_all,         # (V,3,3)
        cam_fovy=cam_fovy_all,   # (V,)
        meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=np.bytes_),
    )
    print(f"[DONE] wrote {out_npz}  (N={images_np.shape[0]}, V={V}, S={S}, DOF={angles_np.shape[1]})")


if __name__ == "__main__":
    main()
'''
python tdcr_to_selfsim_multiview_npz.py \
  --root 2m_no_base \
  --xml tdcr2_no_base.xml \
  --out_npz sim_2m_no_base.npz \
  --size 100 \
  --normalize_ctrl ctrlrange01 \
  --clip_ctrl \
  --strict \
  --auto_near_far \
  --pc_dir 2m_no_base/pointcloud \
  --pc_ext .ply \
  --nf_per_cloud_qmin 1 \
  --nf_per_cloud_qmax 99 \
  --nf_far_quantile 99.5 \
  --nf_margin 0.02 \
  --nf_near_floor 0.01

python tdcr_to_selfsim_multiview_npz.py \
  --root 2m_with_base \
  --xml tdcr2_with_base.xml \
  --out_npz sim_2m_with_base.npz \
  --size 100 \
  --normalize_ctrl ctrlrange01 \
  --clip_ctrl \
  --strict \
  --auto_near_far \
  --pc_dir 2m_with_base/pointcloud \
  --pc_ext .ply \
  --nf_per_cloud_qmin 1 \
  --nf_per_cloud_qmax 99 \
  --nf_far_quantile 99.5 \
  --nf_margin 0.02 \
  --nf_near_floor 0.01


python tdcr_to_selfsim_multiview_npz.py \
  --root 3m_no_base \
  --xml tdcr3_no_base.xml \
  --out_npz sim_3m_no_base.npz \
  --size 100 \
  --normalize_ctrl ctrlrange01 \
  --clip_ctrl \
  --strict \
  --auto_near_far \
  --pc_dir 3m_no_base/pointcloud \
  --pc_ext .ply \
  --nf_per_cloud_qmin 1 \
  --nf_per_cloud_qmax 99 \
  --nf_far_quantile 99.5 \
  --nf_margin 0.02 \
  --nf_near_floor 0.01


python tdcr_to_selfsim_multiview_npz.py \
  --root 3m_with_base \
  --xml tdcr3_with_base.xml \
  --out_npz sim_3m_with_base.npz \
  --size 100 \
  --normalize_ctrl ctrlrange01 \
  --clip_ctrl \
  --strict \
  --auto_near_far \
  --pc_dir 3m_with_base/pointcloud \
  --pc_ext .ply \
  --nf_per_cloud_qmin 1 \
  --nf_per_cloud_qmax 99 \
  --nf_far_quantile 99.5 \
  --nf_margin 0.02 \
  --nf_near_floor 0.01


python tdcr_to_selfsim_multiview_npz.py \
  --root 5m_no_base \
  --xml tdcr5_no_base.xml \
  --out_npz sim_5m_no_base.npz \
  --size 100 \
  --normalize_ctrl ctrlrange01 \
  --clip_ctrl \
  --strict \
  --auto_near_far \
  --pc_dir 5m_no_base/pointcloud \
  --pc_ext .ply \
  --nf_per_cloud_qmin 1 \
  --nf_per_cloud_qmax 99 \
  --nf_far_quantile 99.5 \
  --nf_margin 0.02 \
  --nf_near_floor 0.01


python tdcr_to_selfsim_multiview_npz.py \
  --root 5m_with_base \
  --xml tdcr5_with_base.xml \
  --out_npz sim_5m_with_base.npz \
  --size 100 \
  --normalize_ctrl ctrlrange01 \
  --clip_ctrl \
  --strict \
  --auto_near_far \
  --pc_dir 5m_with_base/pointcloud \
  --pc_ext .ply \
  --nf_per_cloud_qmin 1 \
  --nf_per_cloud_qmax 99 \
  --nf_far_quantile 99.5 \
  --nf_margin 0.02 \
  --nf_near_floor 0.01
'''