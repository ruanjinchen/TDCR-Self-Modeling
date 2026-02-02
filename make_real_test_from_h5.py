#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_real_test_from_h5.py

从 **真实世界 TDCR 的 H5 数据集(test split)** 随机抽取若干条样本，
导出为“和你之前 sim/test 小demo一致”的目录结构：

输出结构（以 real_2m_with_base 为例）：
  <out_root>/real_2m_with_base/
    motor/000001.json
    motor/000002.json
    motor/000003.json
    pointcloud/000001.ply
    pointcloud/000002.ply
    pointcloud/000003.ply
    (可选) pointcloud_norm/000001.ply ...
    selected_meta.json
    (可选) global_motors_scope-all.npz / global_norm_scope-all_anchor-origin.json（复制自原数据集）

motor json 格式对齐你之前的小demo：
  {
    "ctrl": [...],        # motors_norm (0..1)
    "ctrl_raw": [...],    # motors raw
    "actuator_names": ["motor1", ...],
    "norm": {
      "type": "minmax",
      "key": "all",
      "stats_file": "/abs/path/to/global_motors_scope-all.npz",
      "min": [...],
      "max": [...]
    },
    "source": {...}
  }

点云：默认导出 **raw/world 坐标系** 的 GT（带 RGB，如果 H5 里有）。

用法示例：
  python make_real_test_from_h5.py \
    --datasets \
      /data/yxk/K-data/K/fllm-sm/datasets/real/2m_with_base \
      /data/yxk/K-data/K/fllm-sm/datasets/real/3m_with_base \
    --out_root /data/yxk/K-data/K/fllm-sm/sim/test \
    --k 3 \
    --seed 123 \
    --copy_global_files \
    --save_norm_ply

注意：
- 你原始 real 数据集目录名是 2m_with_base / 3m_with_base；本脚本会自动加上 real_ 前缀。
- 如需从 000000 开始编号，设置 --start_id 0。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py  # type: ignore
except Exception as e:
    raise RuntimeError("需要 h5py: pip install h5py") from e


# ------------------------- small utils -------------------------

def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_uint8_rgb(rgb_like: np.ndarray) -> np.ndarray:
    """Convert rgb array to uint8 [0,255]. Accept float [0,1] or float/int [0,255]."""
    c = np.asarray(rgb_like)
    if c.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"RGB array must be (N,3), got {c.shape}")

    if c.dtype == np.uint8:
        return c

    cf = c.astype(np.float32, copy=False)
    mx = float(np.nanmax(cf)) if cf.size else 0.0
    if mx <= 1.0 + 1e-6:
        cf = cf * 255.0
    return np.clip(np.round(cf), 0.0, 255.0).astype(np.uint8)


def save_ply_xyzrgb_ascii(path: Path, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    """Write ASCII PLY. If rgb provided, write uchar RGB."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"xyz must be (N,>=3), got {xyz.shape}")
    xyz = xyz[:, :3]

    has_rgb = rgb is not None
    if has_rgb:
        rgb_u8 = to_uint8_rgb(np.asarray(rgb))
        if rgb_u8.shape[0] != xyz.shape[0]:
            raise ValueError(f"rgb rows {rgb_u8.shape[0]} != xyz rows {xyz.shape[0]}")
    else:
        rgb_u8 = None

    mkdir(path.parent)
    n = int(xyz.shape[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_rgb:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if has_rgb:
            for p, c in zip(xyz, rgb_u8):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def _first_existing_key(f: h5py.File, candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in f:
            return str(k)
    return None


def load_global_motor_minmax(npz_path: Path, key: str = "all") -> Tuple[np.ndarray, np.ndarray]:
    """Load per-dim motor min/max from global_motors_scope-all.npz."""
    obj = np.load(str(npz_path))
    keys = obj["keys"]
    # keys might be np.str_ or bytes
    keys_list = [k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k) for k in keys.tolist()]
    if key not in keys_list:
        raise KeyError(f"Key '{key}' not found in {npz_path}. available={keys_list}")
    i = keys_list.index(key)
    mins = np.asarray(obj["mins"][i], dtype=np.float32).reshape(-1)
    maxs = np.asarray(obj["maxs"][i], dtype=np.float32).reshape(-1)
    return mins, maxs


def compute_motor_norm(ctrl_raw: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1] (keep NaNs)."""
    x = np.asarray(ctrl_raw, dtype=np.float32).reshape(-1)
    mn = np.asarray(mn, dtype=np.float32).reshape(-1)
    mx = np.asarray(mx, dtype=np.float32).reshape(-1)
    if x.shape[0] != mn.shape[0] or x.shape[0] != mx.shape[0]:
        raise ValueError(f"motor dim mismatch: raw={x.shape[0]} min={mn.shape[0]} max={mx.shape[0]}")
    scale = mx - mn
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)

    out = np.full_like(x, np.nan, dtype=np.float32)
    valid = np.isfinite(x) & (~np.isnan(x))
    out[valid] = (x[valid] - mn[valid]) / scale[valid]
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def list_test_refs(dataset_root: Path) -> List[Tuple[Path, int]]:
    """Return list of (shard_path, row_index) for dataset_root/test/shard-*.h5"""
    test_dir = dataset_root / "test"
    shards = sorted(test_dir.glob("shard-*.h5"))
    if not shards:
        raise FileNotFoundError(f"No test shards found under: {test_dir}")

    refs: List[Tuple[Path, int]] = []
    for sp in shards:
        with h5py.File(str(sp), "r") as f:
            # try find the point dataset
            key_data = _first_existing_key(f, ["data", "points", "pointcloud", "pc", "xyz"])
            if key_data is None:
                raise KeyError(f"Cannot find point dataset key in {sp}. keys={list(f.keys())}")
            B = int(f[key_data].shape[0])
        refs.extend([(sp, i) for i in range(B)])
    return refs


def sample_refs(refs: List[Tuple[Path, int]], k: int, seed: int) -> List[Tuple[Path, int]]:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if not refs:
        return []
    k = min(int(k), len(refs))
    rng = np.random.default_rng(int(seed) % (2**32))
    idx = rng.choice(len(refs), size=k, replace=False)
    return [refs[int(i)] for i in idx]


def export_one_dataset(
    dataset_root: Path,
    out_root: Path,
    k: int,
    seed: int,
    start_id: int,
    copy_global_files: bool,
    save_norm_ply: bool,
) -> Path:
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    # output name: real_<basename>
    out_name = f"real_{dataset_root.name}"
    out_dir = (out_root / out_name).resolve()

    motor_dir = out_dir / "motor"
    pc_dir = out_dir / "pointcloud"
    pc_norm_dir = out_dir / "pointcloud_norm"

    mkdir(motor_dir)
    mkdir(pc_dir)
    if save_norm_ply:
        mkdir(pc_norm_dir)

    # copy global files for convenience
    stats_src = dataset_root / "global_motors_scope-all.npz"
    normjson_src = dataset_root / "global_norm_scope-all_anchor-origin.json"

    stats_path = stats_src
    normjson_path = normjson_src

    if copy_global_files:
        if stats_src.exists():
            stats_dst = out_dir / stats_src.name
            shutil.copy2(stats_src, stats_dst)
            # also copy json if exists
            js = dataset_root / "global_motors_scope-all.json"
            if js.exists():
                shutil.copy2(js, out_dir / js.name)
            stats_path = stats_dst
        if normjson_src.exists():
            normjson_dst = out_dir / normjson_src.name
            shutil.copy2(normjson_src, normjson_dst)
            # also copy npz if exists
            npz2 = dataset_root / "global_norm_scope-all_anchor-origin.npz"
            if npz2.exists():
                shutil.copy2(npz2, out_dir / npz2.name)
            normjson_path = normjson_dst

    # load motor min/max
    if not stats_path.exists():
        raise FileNotFoundError(
            f"global motor stats not found: {stats_path}. "
            "Expected dataset_root/global_motors_scope-all.npz"
        )
    mn, mx = load_global_motor_minmax(stats_path, key="all")

    # sample refs
    refs = list_test_refs(dataset_root)
    sel = sample_refs(refs, k=k, seed=seed)

    selected_records: List[Dict[str, object]] = []

    # candidate keys
    key_data_candidates = ["data", "points", "pointcloud", "pc", "xyz"]
    key_data_norm_candidates = ["data_norm", "points_norm", "pointcloud_norm", "pc_norm"]
    key_rgb_candidates = ["rgb", "colors", "color"]
    key_motor_candidates = ["motors", "motor", "ctrl"]
    key_motor_norm_candidates = ["motors_norm", "motor_norm", "ctrl_norm"]

    for j, (shard_path, row) in enumerate(sel):
        out_id = int(start_id) + j
        stem = f"{out_id:06d}"

        with h5py.File(str(shard_path), "r") as f:
            key_data = _first_existing_key(f, key_data_candidates)
            if key_data is None:
                raise KeyError(f"No point dataset found in {shard_path}. keys={list(f.keys())}")
            arr = np.asarray(f[key_data][row])

            # points may be (N,3) or (N,6)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(f"{shard_path}:{key_data}[{row}] has bad shape {arr.shape}")
            xyz_raw = arr[:, :3].astype(np.float32, copy=False)
            rgb = None
            if arr.shape[1] >= 6:
                rgb = arr[:, 3:6]
            else:
                key_rgb = _first_existing_key(f, key_rgb_candidates)
                if key_rgb is not None:
                    rgb = np.asarray(f[key_rgb][row])

            # normalized points (optional)
            xyz_norm = None
            key_dn = _first_existing_key(f, key_data_norm_candidates)
            if key_dn is not None:
                dn = np.asarray(f[key_dn][row])
                if dn.ndim == 2 and dn.shape[1] >= 3:
                    xyz_norm = dn[:, :3].astype(np.float32, copy=False)

            # motors
            key_m = _first_existing_key(f, key_motor_candidates)
            if key_m is None:
                raise KeyError(f"No motor dataset found in {shard_path}. keys={list(f.keys())}")
            ctrl_raw = np.asarray(f[key_m][row], dtype=np.float32).reshape(-1)

            key_mn = _first_existing_key(f, key_motor_norm_candidates)
            if key_mn is not None:
                ctrl = np.asarray(f[key_mn][row], dtype=np.float32).reshape(-1)
            else:
                ctrl = compute_motor_norm(ctrl_raw, mn, mx)

        # write pointcloud
        save_ply_xyzrgb_ascii(pc_dir / f"{stem}.ply", xyz_raw, rgb=rgb)
        if save_norm_ply and xyz_norm is not None:
            save_ply_xyzrgb_ascii(pc_norm_dir / f"{stem}.ply", xyz_norm, rgb=rgb)

        # write motor json
        D = int(ctrl_raw.shape[0])
        motor_obj: Dict[str, object] = {
            "ctrl": [float(x) for x in np.asarray(ctrl, dtype=np.float32).reshape(-1).tolist()],
            "ctrl_raw": [float(x) for x in np.asarray(ctrl_raw, dtype=np.float32).reshape(-1).tolist()],
            "actuator_names": [f"motor{i+1}" for i in range(D)],
            "norm": {
                "type": "minmax",
                "key": "all",
                "stats_file": str(stats_path.resolve()),
                "min": [float(x) for x in mn.tolist()],
                "max": [float(x) for x in mx.tolist()],
            },
            "source": {
                "dataset_root": str(dataset_root),
                "split": "test",
                "shard": shard_path.name,
                "row": int(row),
            },
        }
        with open(motor_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(motor_obj, f, indent=2)

        selected_records.append(
            {
                "out_id": out_id,
                "stem": stem,
                "src_shard": str(shard_path),
                "src_row": int(row),
                "npoints": int(xyz_raw.shape[0]),
                "has_rgb": bool(rgb is not None),
                "motor_dim": int(ctrl_raw.shape[0]),
            }
        )

    meta = {
        "dataset_root": str(dataset_root),
        "split": "test",
        "out_dir": str(out_dir),
        "k": int(k),
        "seed": int(seed),
        "start_id": int(start_id),
        "copy_global_files": bool(copy_global_files),
        "save_norm_ply": bool(save_norm_ply),
        "global_motors_npz": str(stats_path.resolve()) if stats_path.exists() else None,
        "global_norm_json": str(normjson_path.resolve()) if normjson_path.exists() else None,
        "samples": selected_records,
    }
    with open(out_dir / "selected_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] {out_name}: exported {len(selected_records)} samples -> {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("make_real_test_from_h5")
    ap.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        required=True,
        help="真实数据集根目录列表（每个目录应包含 test/shard-*.h5 和 global_* 文件）",
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="输出根目录（会在里面创建 real_<name>/...）",
    )
    ap.add_argument("--k", type=int, default=3, help="每个数据集抽取多少条（默认 3）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    ap.add_argument(
        "--start_id",
        type=int,
        default=1,
        help="导出编号起始值（默认 1 -> 000001；如果想从 000000 开始就设 0）",
    )
    ap.add_argument(
        "--copy_global_files",
        action="store_true",
        help="把 global_motors_scope-all.* 和 global_norm_scope-all_anchor-origin.* 复制到输出目录里",
    )
    ap.add_argument(
        "--save_norm_ply",
        action="store_true",
        help="额外导出归一化坐标系下的 GT 点云到 pointcloud_norm/（如果 H5 里有 data_norm）",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    out_root = args.out_root
    mkdir(out_root)

    # 每个数据集用不同 seed，避免两个集合抽到"看起来一样"的编号分布
    base_seed = int(args.seed)

    for i, ds_root in enumerate(args.datasets):
        export_one_dataset(
            dataset_root=Path(ds_root),
            out_root=out_root,
            k=int(args.k),
            seed=base_seed + i * 1000,
            start_id=int(args.start_id),
            copy_global_files=bool(args.copy_global_files),
            save_norm_ply=bool(args.save_norm_ply),
        )


if __name__ == "__main__":
    main()
'''
python make_real_test_from_h5.py \
  --datasets \
    /data/yxk/K-data/K/fllm-sm/datasets/real/2m_with_base \
    /data/yxk/K-data/K/fllm-sm/datasets/real/3m_with_base \
  --out_root /data/yxk/K-data/K/fllm-sm/sim/test \
  --k 3 \
  --seed 42 \
  --copy_global_files


'''