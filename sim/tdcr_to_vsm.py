#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDCR self-modeling data  ->  VSM (visual-selfmodeling) dataset maker

This script converts TDCR pipeline outputs into the dataset format expected by your VSM code
(BoyuanChen/visual-selfmodeling, MultipleModel dataset).

VSM expects (in one folder, e.g. data/tdcr_2m_no_base/):
  - mesh_<id>.xyzn.npy   (recommended)  or mesh_<id>.xyzn (txt)
      Each row: x y z nx ny nz
  - robot_state.json
      Dict[str(id)] -> [[s0],[s1],...]
      NOTE: your MultipleModel loader does: state = robot_state[k][0] / pi
            Therefore, by default we store states in [-pi, pi] so that /pi -> [-1,1].

And VSM also expects (in your VSM repo root):
  - assets/datainfo/multiple_models_data_split_dict_<seed>.json
      {"train":[...], "test":[...]}

Normal orientation:
  - We first estimate normals by Open3D (unoriented).
  - Then we enforce local consistency (no checkerboard flips) using:
      * Open3D orient_normals_consistent_tangent_plane(k) if available,
        else a kNN BFS sign propagation fallback.
  - Finally we do ONE global flip (all normals * -1) so that normals are
    consistently "outward" or "inward" w.r.t. a prior:
      * prior=camera: vector v = p - nearest_camera_position
      * prior=center: v = p - mean(p)
      * prior=origin: v = p

Why global flip only?
  Per-point flipping using camera priors can re-introduce local sign noise,
  which is harmful for VSM (it uses normals heavily in the loss).

Camera poses:
  Simulation:
    --xml <mujoco.xml>  -> static camera positions read from MuJoCo data.cam_xpos
  Real robot:
    --cam_icp_yaml_dir <dir> with OpenCV YAML files like:
      108322072580_to_243222074200_icp.yaml
    Those files are OpenCV FileStorage YAML (%YAML:1.0, !!opencv-matrix) and should be
    parsed with cv2.FileStorage (PyYAML will fail).
    We build a camera graph and compute all camera positions in a chosen root camera frame.

Assumptions for real data:
  The exported point clouds are already expressed in the same coordinate frame as
  the chosen --cam_root_sn camera (i.e., you transformed/merged all camera clouds
  into that root frame). If your point clouds are in another world frame, you must
  provide that extra rigid transform (not handled here).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional deps
try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mujoco  # type: ignore
except Exception:
    mujoco = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ------------------------------
# Small utils
# ------------------------------
def _print(*a):
    print(*a, flush=True)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_int_id(p: Path) -> int:
    """Parse sample id from filename like 000001.ply or mesh_1.xyzn."""
    m = re.search(r"(\d+)", p.stem)
    if not m:
        raise ValueError(f"Cannot parse id from {p.name}")
    return int(m.group(1))


def list_pc_files(pc_dir: Path) -> List[Path]:
    if not pc_dir.exists():
        raise FileNotFoundError(pc_dir)
    exts = [".ply", ".pcd", ".xyz", ".xyzn", ".npz", ".npy"]
    out = []
    for e in exts:
        out += sorted(pc_dir.glob(f"*{e}"))
    # prefer ply/pcd; if both exist, keep them all but user should be careful
    return out


# ------------------------------
# TDCR loaders
# ------------------------------
def load_point_cloud_xyz(path: Path) -> np.ndarray:
    """Load xyz from a point cloud file."""
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(str(path))
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Bad npy shape: {arr.shape} in {path}")
        return arr[:, :3].astype(np.float32, copy=False)

    if suffix == ".npz":
        obj = np.load(str(path))
        # common keys
        for k in ["xyz", "points", "pc", "arr_0"]:
            if k in obj:
                arr = obj[k]
                arr = np.asarray(arr)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3].astype(np.float32, copy=False)
        raise ValueError(f"Unsupported npz keys: {list(obj.keys())} in {path}")

    if o3d is None:
        raise RuntimeError("open3d is required to read .ply/.pcd/.xyz point clouds.")
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    return xyz


def voxel_downsample_xyz(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size is None or voxel_size <= 0 or len(xyz) == 0:
        return xyz
    if o3d is None:
        raise RuntimeError("open3d is required for voxel downsample.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd2 = pcd.voxel_down_sample(float(voxel_size))
    return np.asarray(pcd2.points, dtype=np.float32)


def random_downsample_xyz(xyz: np.ndarray, npoints: Optional[int], seed: int) -> np.ndarray:
    if npoints is None or npoints <= 0 or len(xyz) <= npoints:
        return xyz
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(xyz), size=int(npoints), replace=False)
    return xyz[idx].astype(np.float32, copy=False)


def maybe_crop_base_z(xyz: np.ndarray, base_z: Optional[float]) -> np.ndarray:
    if base_z is None or len(xyz) == 0:
        return xyz
    m = xyz[:, 2] > float(base_z)
    return xyz[m].astype(np.float32, copy=False)


def load_motor_json(path: Path) -> np.ndarray:
    """Load motor control vector from TDCR pipeline json.
    Compatible with:
      - list/tuple  -> [v0, v1, ...]
      - dict with "ctrl"
      - legacy dict with "motor1"... keys
    """
    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj, dtype=np.float32).reshape(-1)
        return arr

    if isinstance(obj, dict):
        if "ctrl" in obj:
            arr = np.asarray(obj["ctrl"], dtype=np.float32).reshape(-1)
            return arr
        # legacy
        keys = [k for k in obj.keys() if re.match(r"motor\d+", str(k))]
        if keys:
            keys = sorted(keys, key=lambda s: int(re.findall(r"\d+", s)[0]))
            arr = np.asarray([obj[k] for k in keys], dtype=np.float32).reshape(-1)
            return arr

    raise ValueError(f"Unsupported motor json format: {path}")


# ------------------------------
# VSM-format writers
# ------------------------------
def save_xyzn_txt(path_xyzn: Path, xyz: np.ndarray, normals: np.ndarray):
    ensure_dir(path_xyzn.parent)
    arr = np.concatenate([xyz, normals], axis=1).astype(np.float32, copy=False)
    with open(path_xyzn, "w") as f:
        for row in arr:
            f.write(
                f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f} "
                f"{row[3]:.8f} {row[4]:.8f} {row[5]:.8f}\n"
            )


def save_xyzn_npy(path_xyzn: Path, xyz: np.ndarray, normals: np.ndarray):
    """Save as <mesh_i.xyzn>.npy, which your MultipleModel loader will load directly."""
    ensure_dir(path_xyzn.parent)
    arr = np.concatenate([xyz, normals], axis=1).astype(np.float32, copy=False)
    np.save(str(path_xyzn) + ".npy", arr)


# ------------------------------
# Global normalization helper
# ------------------------------
def load_global_norm_json(path: Path) -> Tuple[np.ndarray, float]:
    """Read center/scale from tdcr norm_stage dump json."""
    with open(path, "r") as f:
        obj = json.load(f)

    # Expected: {"all": {"center":[...], "scale": ...}, ...}
    # We prefer key "all", fallback to first key.
    if isinstance(obj, dict):
        if "all" in obj:
            c = np.asarray(obj["all"]["center"], dtype=np.float32)
            s = float(obj["all"]["scale"])
            return c.reshape(3), s
        k0 = next(iter(obj.keys()))
        c = np.asarray(obj[k0]["center"], dtype=np.float32)
        s = float(obj[k0]["scale"])
        return c.reshape(3), s
    raise ValueError(f"Unsupported global norm json: {path}")


def dump_global_norm_json(path: Path, center: np.ndarray, scale: float, meta: Optional[Dict] = None) -> None:
    """Dump center/scale to a JSON file that this script can read via --global_norm_json.

    The structure is compatible with :func:`load_global_norm_json`.

    Output format (minimal):
      {
        "all": {
          "center": [cx, cy, cz],
          "scale": 0.123
        }
      }

    You may optionally include an extra top-level "meta" dict; it will be ignored by
    :func:`load_global_norm_json` because we always look for the "all" key.
    """
    ensure_dir(path.parent)
    obj: Dict = {
        "all": {
            "center": [float(x) for x in np.asarray(center, dtype=np.float32).reshape(3).tolist()],
            "scale": float(scale),
        }
    }
    if meta is not None:
        # keep it JSON-serializable
        obj["meta"] = meta
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_global_scale_origin(pc_paths: List[Path], voxel_size: float = 0.0, npoints_for_scan: int = 20000) -> float:
    """Compute global max radius around origin. (anchor=origin)."""
    if len(pc_paths) == 0:
        return 1.0
    it = pc_paths if tqdm is None else tqdm(pc_paths, desc="[scan global scale]", ncols=120)
    rmax = 0.0
    for p in it:
        xyz = load_point_cloud_xyz(p)
        if voxel_size and voxel_size > 0:
            xyz = voxel_downsample_xyz(xyz, voxel_size)
        xyz = random_downsample_xyz(xyz, npoints_for_scan, seed=parse_int_id(p) + 123)
        if len(xyz) == 0:
            continue
        r = float(np.max(np.linalg.norm(xyz, axis=1)))
        if r > rmax:
            rmax = r
    return max(rmax, 1e-6)


# ------------------------------
# Camera handling
# ------------------------------
def load_camera_positions_from_json(path: Path) -> np.ndarray:
    """Support JSON formats:
      1) {"cam_positions": [[x,y,z], ...]}
      2) {"cameras": [{"pos":[x,y,z]}, ...]}
      3) [[x,y,z], ...]
    """
    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        cams = np.asarray(obj, dtype=np.float32)
        if cams.ndim != 2 or cams.shape[1] != 3:
            raise ValueError("camera json list must be (M,3)")
        return cams

    if isinstance(obj, dict):
        if "cam_positions" in obj:
            cams = np.asarray(obj["cam_positions"], dtype=np.float32)
            if cams.ndim != 2 or cams.shape[1] != 3:
                raise ValueError("cam_positions must be (M,3)")
            return cams
        if "cameras" in obj:
            cams = []
            for c in obj["cameras"]:
                if isinstance(c, dict) and "pos" in c:
                    cams.append(c["pos"])
                else:
                    cams.append(c)
            cams = np.asarray(cams, dtype=np.float32)
            if cams.ndim != 2 or cams.shape[1] != 3:
                raise ValueError("cameras[*].pos must be (3,)")
            return cams

    raise ValueError(f"Unsupported camera json format: {path}")


def load_camera_positions_from_mujoco(xml_path: Path) -> np.ndarray:
    if mujoco is None:
        raise RuntimeError("mujoco is required to read camera positions from XML.")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ncam = int(model.ncam)
    if ncam <= 0:
        raise RuntimeError("No cameras in the MuJoCo model.")
    cams = []
    for cid in range(ncam):
        cams.append(np.array(data.cam_xpos[cid], dtype=np.float32).reshape(3))
    return np.stack(cams, axis=0).astype(np.float32)


def load_camera_positions_per_sample(cam_pose_dir: Path, sample_id: int, zfill: int = 6) -> np.ndarray:
    """Optional: per-sample camera positions, expected file: <cam_pose_dir>/000001.json"""
    p = cam_pose_dir / f"{sample_id:0{zfill}d}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return load_camera_positions_from_json(p)


def _read_opencv_yaml_rt(path: Path) -> np.ndarray:
    """Read an OpenCV FileStorage YAML with keys R (3x3) and t (3x1). Return 4x4 T."""
    if cv2 is None:
        raise RuntimeError("cv2 (opencv-python) is required to read OpenCV YAML.")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open yaml: {path}")
    R = fs.getNode("R").mat()
    t = fs.getNode("t").mat()
    # also support a direct 4x4 matrix key if present
    Tnode = fs.getNode("T")
    fs.release()

    if Tnode is not None and not Tnode.empty():
        T = Tnode.mat()
        T = np.asarray(T, dtype=np.float32)
        if T.shape == (4, 4):
            return T

    if R is None or t is None:
        raise ValueError(f"{path} must contain R and t nodes (opencv-matrix).")
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    t = np.asarray(t, dtype=np.float32).reshape(3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def load_camera_positions_from_icp_yamls(
    yaml_paths: List[Path],
    root_sn: Optional[str] = None,
    sn_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Given yaml files named like <snA>_to_<snB>_icp.yaml containing T_{A->B},
    compute camera origin positions for all involved SNs in the chosen root SN frame.

    Returns:
      cam_positions: (M,3) in root frame
      cam_sns: list[str] aligned with cam_positions
    """
    if len(yaml_paths) == 0:
        raise ValueError("No icp yaml files provided.")

    # Parse edges
    adj: Dict[str, Dict[str, np.ndarray]] = {}
    nodes: set[str] = set()
    edges = []

    for p in yaml_paths:
        m = re.search(r"(\d+)_to_(\d+)", p.name)
        if not m:
            raise ValueError(f"Bad icp yaml filename (need <A>_to_<B>*): {p.name}")
        a, b = m.group(1), m.group(2)
        T_ab = _read_opencv_yaml_rt(p)
        nodes.add(a)
        nodes.add(b)
        adj.setdefault(a, {})[b] = T_ab
        adj.setdefault(b, {})[a] = _inv_T(T_ab)
        edges.append((a, b, p))

    # Choose root SN
    if root_sn is None:
        # heuristic: pick the "to" of the first file (sorted) as root
        p0 = sorted(yaml_paths, key=lambda x: x.name)[0]
        m0 = re.search(r"(\d+)_to_(\d+)", p0.name)
        root_sn = m0.group(2) if m0 else sorted(list(nodes))[0]
        _print(f"[cam_icp] --cam_root_sn not given; using root_sn={root_sn} (heuristic from {p0.name})")

    if root_sn not in nodes:
        raise ValueError(f"root_sn={root_sn} not found in YAML nodes: {sorted(nodes)}")

    # BFS to compute T_root_to_sn
    T_root_to: Dict[str, np.ndarray] = {root_sn: np.eye(4, dtype=np.float32)}
    q = [root_sn]
    while q:
        u = q.pop(0)
        for v, T_u_v in adj.get(u, {}).items():
            if v in T_root_to:
                continue
            T_root_to[v] = T_u_v @ T_root_to[u]
            q.append(v)

    missing = sorted(list(nodes - set(T_root_to.keys())))
    if missing:
        raise RuntimeError(f"Could not reach cameras {missing} from root {root_sn}. Check yaml graph connectivity.")

    # Convert to positions in root frame: pos_sn = origin of sn expressed in root frame
    # If T_root_to_sn maps p_root -> p_sn, then origin_sn in root is (-R^T t)
    pos: Dict[str, np.ndarray] = {}
    for sn, T_r_s in T_root_to.items():
        R = T_r_s[:3, :3]
        t = T_r_s[:3, 3]
        pos[sn] = (-R.T @ t).astype(np.float32)

    # order
    if sn_order is None:
        cam_sns = sorted(list(pos.keys()))
    else:
        # keep only those in pos
        cam_sns = [sn for sn in sn_order if sn in pos]
        # append missing if any
        for sn in sorted(list(pos.keys())):
            if sn not in cam_sns:
                cam_sns.append(sn)

    cam_positions = np.stack([pos[sn] for sn in cam_sns], axis=0).astype(np.float32)
    return cam_positions, cam_sns


def load_camera_positions_from_icp_yaml_dir(
    yaml_dir: Path,
    glob_pat: str = "*_icp.yaml",
    root_sn: Optional[str] = None,
    sn_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    if not yaml_dir.exists():
        raise FileNotFoundError(yaml_dir)
    paths = sorted(yaml_dir.glob(glob_pat))
    if len(paths) == 0:
        raise FileNotFoundError(f"No yaml matched {glob_pat} in {yaml_dir}")
    return load_camera_positions_from_icp_yamls(paths, root_sn=root_sn, sn_order=sn_order)


# ------------------------------
# Normal estimation + orientation
# ------------------------------
def estimate_normals_open3d(xyz: np.ndarray, radius: float, max_nn: int) -> np.ndarray:
    """Estimate normals (unoriented) using Open3D."""
    if o3d is None:
        raise RuntimeError("open3d is required to estimate normals.")
    if len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(radius),
            max_nn=int(max_nn),
        )
    )
    try:
        pcd.normalize_normals()
    except Exception:
        pass
    n = np.asarray(pcd.normals, dtype=np.float32)

    if n.size == 0:
        return np.zeros((len(xyz), 3), dtype=np.float32)
    bad = ~np.isfinite(n).all(axis=1)
    if np.any(bad):
        n[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-9, None)
    return (n / norm).astype(np.float32, copy=False)


def orient_normals_consistent_tangent_plane(xyz: np.ndarray, normals: np.ndarray, k: int) -> np.ndarray:
    """Try Open3D's orient_normals_consistent_tangent_plane(k). If not available, fallback to BFS."""
    if len(xyz) == 0:
        return normals
    if k is None or int(k) <= 0:
        return normals

    if o3d is not None:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
            # Newer Open3D has this method
            pcd.orient_normals_consistent_tangent_plane(int(k))
            n2 = np.asarray(pcd.normals, dtype=np.float32)
            # ensure unit
            norm = np.linalg.norm(n2, axis=1, keepdims=True)
            norm = np.clip(norm, 1e-9, None)
            return (n2 / norm).astype(np.float32, copy=False)
        except Exception:
            pass

    # fallback
    return orient_normals_consistent_knn(xyz, normals, k=k)


def orient_normals_consistent_knn(xyz: np.ndarray, normals: np.ndarray, k: int = 50) -> np.ndarray:
    """Simple sign propagation on kNN graph (fallback).
    This removes local sign flips but cannot decide global in/out.
    """
    if o3d is None:
        # without open3d, implement a slow numpy kNN fallback? (skip)
        return normals
    if len(xyz) == 0:
        return normals
    k = int(max(2, k))

    pts = np.asarray(xyz, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float32).copy()
    # ensure unit
    nn = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.clip(nn, 1e-9, None)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    kdt = o3d.geometry.KDTreeFlann(pcd)

    N = len(pts)
    visited = np.zeros(N, dtype=np.bool_)
    for s in range(N):
        if visited[s]:
            continue
        # start new component
        stack = [s]
        visited[s] = True
        while stack:
            i = stack.pop()
            _, idx, _ = kdt.search_knn_vector_3d(pcd.points[i], k)
            for j in idx:
                if j == i:
                    continue
                if not visited[j]:
                    # make j consistent with i
                    if float(np.dot(nrm[i], nrm[j])) < 0.0:
                        nrm[j] *= -1.0
                    visited[j] = True
                    stack.append(j)
                else:
                    # if already visited, we don't flip to avoid oscillations
                    pass
    return nrm.astype(np.float32, copy=False)


def local_flip_rate(xyz: np.ndarray, normals: np.ndarray, k: int = 20, ncheck: int = 3000, seed: int = 0) -> float:
    """Heuristic metric: fraction of neighbor pairs with dot<0.
    We sample up to ncheck points for speed.
    """
    if o3d is None or len(xyz) == 0:
        return 0.0
    k = int(max(2, k))
    rng = np.random.default_rng(int(seed))
    N = len(xyz)
    sel = np.arange(N)
    if N > ncheck:
        sel = rng.choice(N, size=int(ncheck), replace=False)

    pts = np.asarray(xyz, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float32)
    # unit
    nrm = nrm / np.clip(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9, None)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    kdt = o3d.geometry.KDTreeFlann(pcd)

    bad = 0
    tot = 0
    for i in sel:
        _, idx, _ = kdt.search_knn_vector_3d(pcd.points[int(i)], k)
        ni = nrm[int(i)]
        for j in idx[1:]:
            tot += 1
            if float(np.dot(ni, nrm[int(j)])) < 0.0:
                bad += 1
    return float(bad) / float(max(tot, 1))


def global_orient_normals(
    xyz: np.ndarray,
    normals: np.ndarray,
    prior: str = "camera",
    target: str = "outward",
    cam_positions: Optional[np.ndarray] = None,
    conf_thr: float = 0.05,
) -> Tuple[np.ndarray, Dict]:
    """Flip ALL normals once based on a global vote so that:
      target='outward': dot(n, v) > 0
      target='inward' : dot(n, v) < 0

    prior:
      - camera: v = p - nearest_camera
      - center: v = p - mean(p)
      - origin: v = p
    """
    if len(xyz) == 0:
        return normals, {"flipped": False, "used": 0}

    P = np.asarray(xyz, dtype=np.float32)
    N = np.asarray(normals, dtype=np.float32)
    # unit
    N = N / np.clip(np.linalg.norm(N, axis=1, keepdims=True), 1e-9, None)

    if prior == "camera":
        if cam_positions is None or len(cam_positions) == 0:
            raise ValueError("prior=camera needs cam_positions.")
        cams = np.asarray(cam_positions, dtype=np.float32)
        # nearest camera by distance
        # dist2: (N, M)
        dist2 = np.sum((P[:, None, :] - cams[None, :, :]) ** 2, axis=2)
        j = np.argmin(dist2, axis=1)
        V = P - cams[j]
    elif prior == "center":
        c = np.mean(P, axis=0)
        V = P - c.reshape(1, 3)
    elif prior == "origin":
        V = P
    else:
        raise ValueError(f"Unknown prior: {prior}")

    vnorm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.clip(vnorm, 1e-9, None)

    dot = np.sum(N * V, axis=1)  # (N,)
    mask = np.abs(dot) >= float(conf_thr)
    used = int(np.sum(mask))
    dot_used = dot[mask] if used > 0 else dot

    # vote by median (robust)
    med = float(np.median(dot_used))
    mean = float(np.mean(dot_used))
    pos_frac = float(np.mean(dot_used > 0.0))

    want_positive = (target == "outward")
    # If target=inward, want negative
    if target not in ("outward", "inward"):
        raise ValueError("target must be outward|inward")

    flip = False
    # decide by median sign; if exactly 0, fallback to mean
    s = med if abs(med) > 1e-9 else mean
    if want_positive:
        flip = (s < 0.0)
    else:
        flip = (s > 0.0)

    N2 = (-N if flip else N).astype(np.float32, copy=False)
    info = {
        "prior": prior,
        "target": target,
        "conf_thr": float(conf_thr),
        "used": used,
        "N": int(len(P)),
        "median_dot": med,
        "mean_dot": mean,
        "pos_frac": pos_frac,
        "flipped": bool(flip),
    }
    return N2, info


# ------------------------------
# Visualization helpers
# ------------------------------
def normals_as_lineset(xyz: np.ndarray, normals: np.ndarray, length: float = 0.01, step: int = 20):
    """Create an Open3D LineSet visualizing normals as short segments."""
    if o3d is None:
        raise RuntimeError("open3d is required for visualization.")
    xyz = np.asarray(xyz, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if len(xyz) == 0:
        return o3d.geometry.LineSet()
    step = int(max(1, step))
    pts0 = xyz[::step]
    n0 = normals[::step]
    pts1 = pts0 + n0 * float(length)

    pts = np.vstack([pts0, pts1]).astype(np.float64)
    m = len(pts0)
    lines = np.column_stack([np.arange(m), np.arange(m) + m]).astype(np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    return ls


def visualize_open3d(
    xyz: np.ndarray,
    normals: np.ndarray,
    cam_positions: Optional[np.ndarray] = None,
    normal_len: float = 0.01,
    normal_step: int = 20,
    title: str = "",
):
    if o3d is None:
        raise RuntimeError("open3d is required for visualization.")
    xyz = np.asarray(xyz, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    # store normals for convenience (not used for drawing length)
    if len(normals) == len(xyz):
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    geoms = [pcd]

    # normal lines
    if normal_len is not None and float(normal_len) > 0:
        geoms.append(normals_as_lineset(xyz, normals, length=float(normal_len), step=int(normal_step)))

    # camera markers
    if cam_positions is not None and len(cam_positions) > 0:
        cams = np.asarray(cam_positions, dtype=np.float32)
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = np.linalg.norm(bbox.get_extent())
        r = 0.01 * max(extent, 1e-3)
        for c in cams:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=float(r))
            s.translate(c.astype(np.float64))
            s.compute_vertex_normals()
            geoms.append(s)

    _print(f"[viz] {title}  points={len(xyz)}  normal_len={normal_len}  step={normal_step}")
    try:
        o3d.visualization.draw_geometries(geoms)
    except Exception:
        o3d.visualization.draw_geometries(geoms)


# ------------------------------
# Split + state
# ------------------------------
def make_split(ids: List[int], seed: int, test_frac: float) -> Dict[str, List[int]]:
    ids = list(sorted(ids))
    rng = np.random.default_rng(int(seed))
    perm = np.array(ids, dtype=np.int64)
    rng.shuffle(perm)

    ntest = int(round(len(perm) * float(test_frac)))
    if len(perm) >= 2:
        ntest = max(1, min(ntest, len(perm) - 1))
    else:
        ntest = len(perm)

    test_ids = perm[:ntest].tolist()
    train_ids = perm[ntest:].tolist()
    return {"train": train_ids, "test": test_ids}


def build_robot_state(
    ids: List[int],
    motor_paths: Dict[int, Path],
    do_minmax_to_unit: bool = True,
    pi_scale: bool = True,
) -> Tuple[Dict[str, List[List[float]]], Dict]:
    """Return robot_state_dict and stats.
    - robot_state_dict[str(id)] = [[s0],[s1],...]
    """
    motors = []
    for i in ids:
        motors.append(load_motor_json(motor_paths[i]).reshape(1, -1))
    motors = np.concatenate(motors, axis=0).astype(np.float32)  # (N,D)

    stats = {}
    stats["D"] = int(motors.shape[1])
    stats["min"] = motors.min(axis=0).tolist()
    stats["max"] = motors.max(axis=0).tolist()
    stats["mean"] = motors.mean(axis=0).tolist()

    X = motors
    if do_minmax_to_unit:
        mn = motors.min(axis=0)
        mx = motors.max(axis=0)
        mid = (mn + mx) * 0.5
        half = (mx - mn) * 0.5
        half = np.where(half < 1e-9, 1.0, half)
        X = (motors - mid) / half  # approx [-1,1]
        stats["mid"] = mid.tolist()
        stats["half"] = half.tolist()

    if pi_scale:
        X = X * np.pi

    robot_state: Dict[str, List[List[float]]] = {}
    for idx, sid in enumerate(ids):
        vec = X[idx].tolist()
        robot_state[str(int(sid))] = [[float(v)] for v in vec]
    return robot_state, stats


# ------------------------------
# Main pipeline
# ------------------------------
@dataclass
class BuildCfg:
    pc_dir: Path
    motor_dir: Path
    out_dir: Path
    vsm_root: Path
    seed: int = 42
    test_frac: float = 0.1

    # point cloud processing
    voxel_size: float = 0.0
    npoints: Optional[int] = None
    base_z: Optional[float] = None

    # normalization
    global_norm_json: Optional[Path] = None
    compute_global_scale: bool = False  # if no global_norm_json
    scan_npoints_for_scale: int = 20000

    # normalization overrides / dump
    # If set, these will override whatever comes from --global_norm_json or --compute_global_scale.
    #   xyz_norm = (xyz - center) / scale
    norm_center: Optional[Tuple[float, float, float]] = None
    norm_scale: Optional[float] = None
    dump_global_norm_json: Optional[Path] = None

    # normals
    normal_radius: float = 0.03
    normal_max_nn: int = 40
    normal_consistent_k: int = 80
    normal_prior: str = "camera"     # camera|center|origin
    normal_target: str = "outward"   # outward|inward
    normal_conf_thr: float = 0.05

    # camera static sources
    cam_json: Optional[Path] = None
    xml: Optional[Path] = None
    cam_icp_yaml_dir: Optional[Path] = None
    cam_icp_glob: str = "*_icp.yaml"
    cam_root_sn: Optional[str] = None
    cam_sn_order: Optional[str] = None   # comma-separated
    dump_cam_json: Optional[Path] = None

    # per-sample camera positions
    cam_pose_dir: Optional[Path] = None
    cam_zfill: int = 6

    # output format
    save_txt: bool = False
    save_npy: bool = True
    resume: bool = True

    # states
    state_minmax_to_unit: bool = True
    state_pi_scale: bool = True

    # preview
    preview_k: int = 5
    preview_seed: int = 0
    preview_voxel: float = 0.005
    preview_npoints: int = 6000
    preview_normal_len: float = 0.01
    preview_normal_step: int = 25


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("TDCR -> VSM dataset maker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser):
        p.add_argument("--pc_dir", type=Path, required=True, help="TDCR pointcloud dir (000001.ply ...)")
        p.add_argument("--motor_dir", type=Path, required=True, help="TDCR motor dir (000001.json ...)")
        p.add_argument("--seed", type=int, default=42, help="seed for VSM split json filename")
        p.add_argument("--test_frac", type=float, default=0.1, help="fraction for test split")

        p.add_argument("--voxel_size", type=float, default=0.002, help="voxel size before normals")
        p.add_argument("--npoints", type=int, default=20000, help="downsample to this many points (<=0 keep all)")
        p.add_argument("--base_z", type=float, default=None, help="optional: keep points with z>base_z (world coords)")

        # normalization
        p.add_argument("--global_norm_json", type=Path, default=None,
                       help="tdcr norm_stage dump json (scope-all anchor-origin), contains center/scale")
        p.add_argument("--compute_global_scale", action="store_true",
                       help="if no global_norm_json, scan pointclouds to compute a global scale around origin")
        p.add_argument("--scan_npoints_for_scale", type=int, default=20000)

        # (optional) override center/scale directly, or dump computed results to a file
        p.add_argument(
            "--norm_center",
            type=float,
            nargs=3,
            default=None,
            metavar=("CX", "CY", "CZ"),
            help="override normalization center in raw coords. xyz_norm=(xyz-center)/scale",
        )
        p.add_argument(
            "--norm_scale",
            type=float,
            default=None,
            help="override normalization scale in raw coords. xyz_norm=(xyz-center)/scale",
        )
        p.add_argument(
            "--dump_global_norm_json",
            type=Path,
            default=None,
            help="dump the used normalization (center/scale) to a JSON that --global_norm_json can read later",
        )

        # normals
        p.add_argument("--normal_radius", type=float, default=0.03, help="normal radius (in normalized coords if using global_norm)")
        p.add_argument("--normal_max_nn", type=int, default=40)
        p.add_argument("--normal_consistent_k", type=int, default=80, help="k for consistent tangent plane (0 disables)")
        p.add_argument("--normal_prior", choices=["camera", "center", "origin"], default="camera")
        p.add_argument("--normal_target", choices=["outward", "inward"], default="outward")
        p.add_argument("--normal_conf_thr", type=float, default=0.05, help="confidence threshold |dot| for global flip vote")

        # cameras (static): choose one
        p.add_argument("--xml", type=Path, default=None, help="MuJoCo xml (simulation) -> read camera positions")
        p.add_argument("--cam_json", type=Path, default=None, help="static camera positions JSON")
        p.add_argument("--cam_icp_yaml_dir", type=Path, default=None,
                       help="dir containing OpenCV YAML extrinsics like <snA>_to_<snB>_icp.yaml")
        p.add_argument("--cam_icp_glob", type=str, default="*_icp.yaml")
        p.add_argument("--cam_root_sn", type=str, default=None, help="root camera SN for YAML extrinsics (pointcloud frame)")
        p.add_argument("--cam_sn_order", type=str, default=None, help="comma-separated SN order for debugging/visualization")
        p.add_argument("--dump_cam_json", type=Path, default=None, help="optional: dump computed static cam positions to JSON")

        # cameras (per sample)
        p.add_argument("--cam_pose_dir", type=Path, default=None, help="optional per-sample camera positions dir: 000001.json")
        p.add_argument("--cam_zfill", type=int, default=6)

        # state normalization
        p.add_argument("--state_raw", action="store_true",
                       help="do NOT minmax normalize motors; store raw values (optionally *pi)")
        p.add_argument("--state_no_pi", action="store_true",
                       help="do NOT multiply states by pi before writing robot_state.json")

    # preview
    ap_prev = sub.add_parser("preview", help="randomly visualize K samples (pc + normals)")
    add_common(ap_prev)
    ap_prev.add_argument("--k", type=int, default=5)
    ap_prev.add_argument("--preview_seed", type=int, default=0)
    ap_prev.add_argument("--preview_voxel", type=float, default=0.005, help="extra voxel downsample for preview only")
    ap_prev.add_argument("--preview_npoints", type=int, default=6000)
    ap_prev.add_argument("--normal_len", type=float, default=0.01, help="visual normal length")
    ap_prev.add_argument("--normal_step", type=int, default=25, help="draw one normal each <step> points")

    # build
    ap_build = sub.add_parser("build", help="build full VSM dataset")
    add_common(ap_build)
    ap_build.add_argument("--out_dir", type=Path, required=True, help="output dataset folder for VSM (data_filepath)")
    ap_build.add_argument("--vsm_root", type=Path, default=Path("."), help="VSM repo root (contains assets/datainfo)")
    ap_build.add_argument("--save_txt", action="store_true", help="also save mesh_*.xyzn text (slow & large)")
    ap_build.add_argument("--no_save_npy", action="store_true", help="do NOT save mesh_*.xyzn.npy")
    ap_build.add_argument("--no_resume", action="store_true", help="recompute even if output exists")

    return ap.parse_args()


def build_cfg_from_args(args: argparse.Namespace) -> BuildCfg:
    npoints = None if (getattr(args, "npoints", 0) is None or int(args.npoints) <= 0) else int(args.npoints)

    norm_center = getattr(args, "norm_center", None)
    if norm_center is not None:
        # argparse gives a list[float] for nargs=3
        norm_center = tuple(float(x) for x in norm_center)
        if len(norm_center) != 3:
            raise ValueError("--norm_center must have exactly 3 numbers: CX CY CZ")

    norm_scale = getattr(args, "norm_scale", None)
    if norm_scale is not None:
        norm_scale = float(norm_scale)

    cfg = BuildCfg(
        pc_dir=args.pc_dir,
        motor_dir=args.motor_dir,
        out_dir=getattr(args, "out_dir", None) or Path("./vsm_data"),
        vsm_root=getattr(args, "vsm_root", Path(".")),
        seed=int(args.seed),
        test_frac=float(args.test_frac),
        voxel_size=float(args.voxel_size),
        npoints=npoints,
        base_z=getattr(args, "base_z", None),
        global_norm_json=getattr(args, "global_norm_json", None),
        compute_global_scale=bool(getattr(args, "compute_global_scale", False)),
        scan_npoints_for_scale=int(getattr(args, "scan_npoints_for_scale", 20000)),
        norm_center=norm_center,
        norm_scale=norm_scale,
        dump_global_norm_json=getattr(args, "dump_global_norm_json", None),
        normal_radius=float(getattr(args, "normal_radius", 0.03)),
        normal_max_nn=int(getattr(args, "normal_max_nn", 40)),
        normal_consistent_k=int(getattr(args, "normal_consistent_k", 80)),
        normal_prior=str(getattr(args, "normal_prior", "camera")),
        normal_target=str(getattr(args, "normal_target", "outward")),
        normal_conf_thr=float(getattr(args, "normal_conf_thr", 0.05)),
        cam_json=getattr(args, "cam_json", None),
        xml=getattr(args, "xml", None),
        cam_icp_yaml_dir=getattr(args, "cam_icp_yaml_dir", None),
        cam_icp_glob=str(getattr(args, "cam_icp_glob", "*_icp.yaml")),
        cam_root_sn=getattr(args, "cam_root_sn", None),
        cam_sn_order=getattr(args, "cam_sn_order", None),
        dump_cam_json=getattr(args, "dump_cam_json", None),
        cam_pose_dir=getattr(args, "cam_pose_dir", None),
        cam_zfill=int(getattr(args, "cam_zfill", 6)),
        save_txt=bool(getattr(args, "save_txt", False)),
        save_npy=not bool(getattr(args, "no_save_npy", False)),
        resume=not bool(getattr(args, "no_resume", False)),
        state_minmax_to_unit=not bool(getattr(args, "state_raw", False)),
        state_pi_scale=not bool(getattr(args, "state_no_pi", False)),
        preview_k=int(getattr(args, "k", 5)),
        preview_seed=int(getattr(args, "preview_seed", 0)),
        preview_voxel=float(getattr(args, "preview_voxel", 0.005)),
        preview_npoints=int(getattr(args, "preview_npoints", 6000)),
        preview_normal_len=float(getattr(args, "normal_len", 0.01)),
        preview_normal_step=int(getattr(args, "normal_step", 25)),
    )
    return cfg


def get_static_cam_positions(cfg: BuildCfg) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Return (cam_positions, cam_sns) or (None,None)."""
    # precedence: icp yaml > cam_json > xml
    if cfg.cam_icp_yaml_dir is not None:
        sn_order = None
        if cfg.cam_sn_order:
            sn_order = [s.strip() for s in cfg.cam_sn_order.split(",") if s.strip()]
        cams, sns = load_camera_positions_from_icp_yaml_dir(
            cfg.cam_icp_yaml_dir,
            glob_pat=cfg.cam_icp_glob,
            root_sn=cfg.cam_root_sn,
            sn_order=sn_order,
        )
        if cfg.dump_cam_json is not None:
            ensure_dir(cfg.dump_cam_json.parent)
            with open(cfg.dump_cam_json, "w") as f:
                json.dump({"cam_sns": sns, "cam_positions": cams.tolist()}, f, indent=2)
            _print(f"[cam_icp] dumped static cameras to {cfg.dump_cam_json}")
        return cams, sns

    if cfg.cam_json is not None:
        cams = load_camera_positions_from_json(cfg.cam_json)
        return cams, None

    if cfg.xml is not None:
        cams = load_camera_positions_from_mujoco(cfg.xml)
        return cams, None

    return None, None


def compute_center_scale(cfg: BuildCfg, pc_paths: List[Path]) -> Tuple[np.ndarray, float, Dict]:
    """Return (center, scale, info).

    Default behavior follows tdcr add-norm (anchor=origin): center=[0,0,0].

    Normalization formula used by this script:
      xyz_norm = (xyz_raw - center) / scale

    Priority:
      1) base from --global_norm_json (if given)
      2) else base from --compute_global_scale (if set)
      3) else base is center=0, scale=1
      4) then apply optional overrides: --norm_center / --norm_scale
    """
    info: Dict = {
        "base_source": "default",
        "global_norm_json": None,
        "computed": False,
        "overrides": [],
    }

    # ----- base center/scale -----
    if cfg.global_norm_json is not None:
        c, s = load_global_norm_json(cfg.global_norm_json)
        center = c.astype(np.float32)
        scale = float(s)
        info["base_source"] = "global_norm_json"
        info["global_norm_json"] = str(cfg.global_norm_json)
    else:
        center = np.zeros(3, dtype=np.float32)
        scale = 1.0
        if cfg.compute_global_scale:
            s = compute_global_scale_origin(
                pc_paths,
                voxel_size=float(cfg.voxel_size),
                npoints_for_scan=int(cfg.scan_npoints_for_scale),
            )
            _print(f"[norm] computed global scale around origin: {s:.6f}")
            scale = float(s)
            info["base_source"] = "computed_global_scale_origin"
            info["computed"] = True
            info["voxel_size"] = float(cfg.voxel_size)
            info["scan_npoints_for_scale"] = int(cfg.scan_npoints_for_scale)

    # ----- overrides -----
    if cfg.norm_center is not None:
        c = np.asarray(cfg.norm_center, dtype=np.float32).reshape(3)
        center = c
        info["overrides"].append("center")

    if cfg.norm_scale is not None:
        scale = float(cfg.norm_scale)
        info["overrides"].append("scale")

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization scale={scale}. It must be finite and > 0.")

    return center.astype(np.float32), float(scale), info


def process_one(
    sid: int,
    pc_path: Path,
    center: np.ndarray,
    scale: float,
    cfg: BuildCfg,
    cam_positions_static: Optional[np.ndarray],
    for_preview: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    """Return (xyz_norm, normals, cam_positions_norm_used, info)."""
    xyz = load_point_cloud_xyz(pc_path)
    if cfg.base_z is not None:
        xyz = maybe_crop_base_z(xyz, cfg.base_z)

    # downsample
    vox = cfg.preview_voxel if for_preview else cfg.voxel_size
    if vox and vox > 0:
        xyz = voxel_downsample_xyz(xyz, float(vox))
    npts = cfg.preview_npoints if for_preview else cfg.npoints
    xyz = random_downsample_xyz(xyz, npts, seed=sid + 17)

    # normalize (P - center)/scale
    xyz_norm = (xyz - center.reshape(1, 3)) / float(scale)

    # camera positions (normalized to same space)
    cam_used = None
    if cfg.cam_pose_dir is not None:
        cams = load_camera_positions_per_sample(cfg.cam_pose_dir, sid, zfill=cfg.cam_zfill)
        cam_used = (cams - center.reshape(1, 3)) / float(scale)
    elif cam_positions_static is not None:
        cam_used = (cam_positions_static - center.reshape(1, 3)) / float(scale)

    # normals
    n0 = estimate_normals_open3d(xyz_norm, radius=cfg.normal_radius, max_nn=cfg.normal_max_nn)
    flip0 = local_flip_rate(xyz_norm, n0, k=min(cfg.normal_consistent_k, 30), seed=sid)

    n1 = orient_normals_consistent_tangent_plane(xyz_norm, n0, k=cfg.normal_consistent_k)
    flip1 = local_flip_rate(xyz_norm, n1, k=min(cfg.normal_consistent_k, 30), seed=sid + 1)

    # global flip
    info_orient = {}
    if cfg.normal_prior == "camera":
        if cam_used is None:
            raise RuntimeError("normal_prior=camera but no camera positions are provided. Use --xml/--cam_json/--cam_icp_yaml_dir/--cam_pose_dir")
        n2, info_orient = global_orient_normals(
            xyz_norm, n1, prior="camera", target=cfg.normal_target, cam_positions=cam_used, conf_thr=cfg.normal_conf_thr
        )
    elif cfg.normal_prior == "center":
        n2, info_orient = global_orient_normals(xyz_norm, n1, prior="center", target=cfg.normal_target, conf_thr=cfg.normal_conf_thr)
    elif cfg.normal_prior == "origin":
        n2, info_orient = global_orient_normals(xyz_norm, n1, prior="origin", target=cfg.normal_target, conf_thr=cfg.normal_conf_thr)
    else:
        n2 = n1

    info = {
        "sid": int(sid),
        "pc": pc_path.name,
        "n_points": int(len(xyz_norm)),
        "flip_rate_before": float(flip0),
        "flip_rate_after": float(flip1),
        "global_orient": info_orient,
    }
    return xyz_norm.astype(np.float32), n2.astype(np.float32), cam_used, info


def preview(cfg: BuildCfg):
    if o3d is None:
        raise RuntimeError("open3d is required for preview.")
    pc_paths = list_pc_files(cfg.pc_dir)
    if len(pc_paths) == 0:
        raise FileNotFoundError(f"No pointcloud files found in {cfg.pc_dir}")

    # map id -> path
    pc_map = {parse_int_id(p): p for p in pc_paths}

    motor_paths = sorted(cfg.motor_dir.glob("*.json"))
    motor_map = {parse_int_id(p): p for p in motor_paths}
    ids = sorted(list(set(pc_map.keys()) & set(motor_map.keys())))
    if len(ids) == 0:
        raise RuntimeError("No matched ids between pointclouds and motors.")

    center, scale, norm_info = compute_center_scale(cfg, [pc_map[i] for i in ids])
    _print(f"[norm] center={center.tolist()}  scale={scale:.6f}  info={norm_info}")

    if cfg.dump_global_norm_json is not None:
        meta = {
            **{k: v for k, v in norm_info.items()},
            "pc_dir": str(cfg.pc_dir),
            "formula": "xyz_norm = (xyz_raw - center) / scale; xyz_raw = xyz_norm * scale + center",
        }
        dump_global_norm_json(cfg.dump_global_norm_json, center=center, scale=scale, meta=meta)
        _print(f"[norm] dumped center/scale to {cfg.dump_global_norm_json}")

    cam_static, cam_sns = get_static_cam_positions(cfg)
    if cam_static is not None:
        _print(f"[cam] static cameras: {len(cam_static)}")
        if cam_sns:
            _print(f"[cam] sns: {cam_sns}")

    rng = np.random.default_rng(int(cfg.preview_seed))
    sel = ids if len(ids) <= cfg.preview_k else rng.choice(ids, size=int(cfg.preview_k), replace=False).tolist()
    sel = list(map(int, sel))

    for sid in sel:
        xyz, nrm, cams, info = process_one(
            sid=sid,
            pc_path=pc_map[sid],
            center=center,
            scale=scale,
            cfg=cfg,
            cam_positions_static=cam_static,
            for_preview=True,
        )
        _print(f"[preview] sid={sid}  flip_rate {info['flip_rate_before']:.3f} -> {info['flip_rate_after']:.3f}  global={info['global_orient']}")
        visualize_open3d(
            xyz, nrm,
            cam_positions=cams,
            normal_len=cfg.preview_normal_len,
            normal_step=cfg.preview_normal_step,
            title=f"sid={sid}",
        )


def build(cfg: BuildCfg):
    pc_paths = list_pc_files(cfg.pc_dir)
    if len(pc_paths) == 0:
        raise FileNotFoundError(f"No pointcloud files found in {cfg.pc_dir}")

    pc_map = {parse_int_id(p): p for p in pc_paths}
    motor_paths = sorted(cfg.motor_dir.glob("*.json"))
    motor_map = {parse_int_id(p): p for p in motor_paths}

    ids = sorted(list(set(pc_map.keys()) & set(motor_map.keys())))
    if len(ids) == 0:
        raise RuntimeError("No matched ids between pointclouds and motors.")
    _print(f"[scan] matched ids: {len(ids)}")

    center, scale, norm_info = compute_center_scale(cfg, [pc_map[i] for i in ids])
    _print(f"[norm] center={center.tolist()}  scale={scale:.6f}  info={norm_info}")

    if cfg.dump_global_norm_json is not None:
        meta = {
            **{k: v for k, v in norm_info.items()},
            "pc_dir": str(cfg.pc_dir),
            "formula": "xyz_norm = (xyz_raw - center) / scale; xyz_raw = xyz_norm * scale + center",
        }
        dump_global_norm_json(cfg.dump_global_norm_json, center=center, scale=scale, meta=meta)
        _print(f"[norm] dumped center/scale to {cfg.dump_global_norm_json}")

    cam_static, cam_sns = get_static_cam_positions(cfg)
    if cam_static is not None:
        _print(f"[cam] static cameras: {len(cam_static)}")

    out_dir = ensure_dir(cfg.out_dir)

    # 1) robot_state.json
    robot_state, motor_stats = build_robot_state(
        ids=ids,
        motor_paths=motor_map,
        do_minmax_to_unit=cfg.state_minmax_to_unit,
        pi_scale=cfg.state_pi_scale,
    )
    with open(out_dir / "robot_state.json", "w") as f:
        json.dump(robot_state, f, indent=2)
    with open(out_dir / "robot_state_stats.json", "w") as f:
        json.dump(motor_stats, f, indent=2)
    _print(f"[state] wrote robot_state.json (keys={len(robot_state)}) and robot_state_stats.json (D={motor_stats['D']})")

    # 2) split json for MultipleModel
    split = make_split(ids, seed=cfg.seed, test_frac=cfg.test_frac)
    split_path = cfg.vsm_root / "assets" / "datainfo" / f"multiple_models_data_split_dict_{cfg.seed}.json"
    ensure_dir(split_path.parent)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    _print(f"[split] wrote {split_path}  train={len(split['train'])} test={len(split['test'])}")

    # 3) export each mesh_i.xyzn(.npy)
    it = ids if tqdm is None else tqdm(ids, desc="[build xyzn]", ncols=120)
    for sid in it:
        out_base = out_dir / f"mesh_{sid}.xyzn"
        if cfg.resume and cfg.save_npy and (Path(str(out_base) + ".npy").exists()):
            continue
        if cfg.resume and (not cfg.save_npy) and out_base.exists():
            continue

        xyz, nrm, cams, info = process_one(
            sid=sid,
            pc_path=pc_map[sid],
            center=center,
            scale=scale,
            cfg=cfg,
            cam_positions_static=cam_static,
            for_preview=False,
        )

        if cfg.save_npy:
            save_xyzn_npy(out_base, xyz, nrm)
        if cfg.save_txt:
            save_xyzn_txt(out_base, xyz, nrm)

    _print(f"[done] dataset in {out_dir}")


def main():
    args = parse_args()
    cfg = build_cfg_from_args(args)

    if args.cmd == "preview":
        preview(cfg)
        return
    if args.cmd == "build":
        build(cfg)
        return
    raise ValueError(args.cmd)


if __name__ == "__main__":
    main()

'''
python tdcr_to_vsm.py preview \
  --pc_dir 2m_no_base/pointcloud \
  --motor_dir 2m_no_base/motor \
  --xml tdcr2.xml \
  --global_norm_json 2m_no_base/global_norm_scope-all_anchor-origin.json \
  --k 5 \
  --normal_len 0.01 --normal_step 2


python tdcr_to_vsm.py preview \
  --pc_dir 2m_with_base/pointcloud \
  --motor_dir 2m_with_base/motor \
  --xml tdcr2.xml \
  --global_norm_json 2m_with_base/global_norm_scope-all_anchor-origin.json \
  --k 5 \
  --normal_len 0.01 --normal_step 2
  
python tdcr_to_vsm.py preview \
  --pc_dir 5m_with_base/pointcloud \
  --motor_dir 5m_with_base/motor \
  --xml tdcr2.xml \
  --global_norm_json 5m_with_base/global_norm_scope-all_anchor-origin.json \
  --k 5 \
  --normal_len 0.01 --normal_step 2


  
python tdcr_to_vsm.py build \
  --pc_dir 2m_with_base/pointcloud \
  --motor_dir 2m_with_base/motor \
  --xml tdcr2.xml \
  --global_norm_json 2m_with_base/global_norm_scope-all_anchor-origin.json \
  --out_dir data/tdcr_2m_with_base_vsm \
  --vsm_root . \
  --seed 44 \
  --test_frac 0.1 \
  --voxel_size 0.002 \
  --npoints 20000

python tdcr_to_vsm.py build \
  --pc_dir 2m_no_base/pointcloud \
  --motor_dir 2m_no_base/motor \
  --xml tdcr2_no_base.xml \
  --global_norm_json 2m_no_base/global_norm_scope-all_anchor-origin.json \
  --out_dir vsm/tdcr_2m_no_base_vsm \
  --vsm_root . \
  --seed 44 \
  --test_frac 0.1 \
  --voxel_size 0.002 \
  --npoints 20000

python tdcr_to_vsm.py build \
  --pc_dir 3m_with_base/pointcloud \
  --motor_dir 3m_with_base/motor \
  --xml tdcr3_with_base.xml \
  --global_norm_json 3m_with_base/global_norm_scope-all_anchor-origin.json \
  --out_dir data/tdcr_3m_with_base_vsm \
  --vsm_root . \
  --seed 43 \
  --test_frac 0.1 \
  --voxel_size 0.002 \
  --npoints 20000

python tdcr_to_vsm.py build \
  --pc_dir 3m_no_base/pointcloud \
  --motor_dir 3m_no_base/motor \
  --xml tdcr3_no_base.xml \
  --global_norm_json 3m_no_base/global_norm_scope-all_anchor-origin.json \
  --out_dir data/tdcr_3m_no_base_vsm \
  --vsm_root . \
  --seed 43 \
  --test_frac 0.1 \
  --voxel_size 0.002 \
  --npoints 20000
  
  
python tdcr_to_vsm.py build \
  --pc_dir 5m_with_base/pointcloud \
  --motor_dir 5m_with_base/motor \
  --xml tdcr5_with_base.xml \
  --global_norm_json 5m_with_base/global_norm_scope-all_anchor-origin.json \
  --out_dir data/tdcr_5m_with_base_vsm \
  --vsm_root . \
  --seed 43 \
  --test_frac 0.1 \
  --voxel_size 0.002 \
  --npoints 20000


python tdcr_to_vsm.py build \
  --pc_dir 5m_no_base/pointcloud \
  --motor_dir 5m_no_base/motor \
  --xml tdcr5_no_base.xml \
  --global_norm_json 5m_no_base/global_norm_scope-all_anchor-origin.json \
  --out_dir data/tdcr_5m_no_base_vsm \
  --vsm_root . \
  --seed 43 \
  --test_frac 0.1 \
  --voxel_size 0.002 \
  --npoints 20000
  
输出vsm的scale
python tdcr_to_vsm.py preview \
  --pc_dir "/data/yxk/K-data/K/fllm-sm/sim/2m_no_base/pointcloud" \
  --motor_dir "/data/yxk/K-data/K/fllm-sm/sim/2m_no_base/motor" \
  --compute_global_scale \
  --voxel_size 0.002 \
  --scan_npoints_for_scale 20000 \
  --dump_global_norm_json "vsm_scale/2m_no_base/global_norm_scope-all_anchor-origin.json" \
  --k 0

python tdcr_to_vsm.py preview \
  --pc_dir "/data/yxk/K-data/K/fllm-sm/sim/2m_with_base/pointcloud" \
  --motor_dir "/data/yxk/K-data/K/fllm-sm/sim/2m_with_base/motor" \
  --compute_global_scale \
  --voxel_size 0.002 \
  --scan_npoints_for_scale 20000 \
  --dump_global_norm_json "vsm_scale/2m_with_base/global_norm_scope-all_anchor-origin.json" \
  --k 0

python tdcr_to_vsm.py preview \
  --pc_dir "/data/yxk/K-data/K/fllm-sm/sim/3m_no_base/pointcloud" \
  --motor_dir "/data/yxk/K-data/K/fllm-sm/sim/3m_no_base/motor" \
  --compute_global_scale \
  --voxel_size 0.002 \
  --scan_npoints_for_scale 20000 \
  --dump_global_norm_json "vsm_scale/3m_no_base/global_norm_scope-all_anchor-origin.json" \
  --k 0

python tdcr_to_vsm.py preview \
  --pc_dir "/data/yxk/K-data/K/fllm-sm/sim/3m_with_base/pointcloud" \
  --motor_dir "/data/yxk/K-data/K/fllm-sm/sim/3m_with_base/motor" \
  --compute_global_scale \
  --voxel_size 0.002 \
  --scan_npoints_for_scale 20000 \
  --dump_global_norm_json "vsm_scale/3m_with_base/global_norm_scope-all_anchor-origin.json" \
  --k 0

python tdcr_to_vsm.py preview \
  --pc_dir "/data/yxk/K-data/K/fllm-sm/sim/5m_no_base/pointcloud" \
  --motor_dir "/data/yxk/K-data/K/fllm-sm/sim/5m_no_base/motor" \
  --compute_global_scale \
  --voxel_size 0.002 \
  --scan_npoints_for_scale 20000 \
  --dump_global_norm_json "vsm_scale/5m_no_base/global_norm_scope-all_anchor-origin.json" \
  --k 0

python tdcr_to_vsm.py preview \
  --pc_dir "/data/yxk/K-data/K/fllm-sm/sim/5m_with_base/pointcloud" \
  --motor_dir "/data/yxk/K-data/K/fllm-sm/sim/5m_with_base/motor" \
  --compute_global_scale \
  --voxel_size 0.002 \
  --scan_npoints_for_scale 20000 \
  --dump_global_norm_json "vsm_scale/5m_with_base/global_norm_scope-all_anchor-origin.json" \
  --k 0


'''