#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --- make sure we can import "tdcr" package under sim/ ---
SIM_DIR = Path(__file__).resolve().parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from tdcr.utils import ensure_dir, print_color
from tdcr import collect as tdcr_collect
from tdcr import norm as tdcr_norm

try:
    import h5py
except Exception:
    h5py = None


def find_dataset_root(p: Path) -> Path:
    """
    用户可能传：root目录 / train/shard-xxxx.h5 / global_motors_xxx.npz
    这里做一个向上探测，找出真正的数据集 root（包含 train/val/test 或 global_motors_*.npz）。
    """
    p = p.expanduser().resolve()
    candidates = []
    if p.is_file():
        candidates = [p.parent, p.parent.parent, p.parent.parent.parent]
    else:
        candidates = [p, p.parent, p.parent.parent]

    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        has_split = any((d / sp).is_dir() for sp in ("train", "val", "test"))
        has_global = any(d.glob("global_motors_*.npz")) or any(d.glob("global_motors_*.json"))
        if has_split or has_global:
            return d
    return p if p.is_dir() else p.parent


def infer_xml(dataset_root: Path, explicit: str | None) -> Path:
    """
    不传 --xml 时，按你 cmd.txt 的目录命名约定自动推断：
      2m_no_base  -> tdcr2_no_base.xml
      2m_with_base-> tdcr2_with_base.xml
      3m_no_base  -> tdcr3_no_base.xml
      ...
    找不到就要求你显式传 --xml。
    """
    if explicit:
        xp = Path(explicit)
        if not xp.is_absolute():
            # try relative to cwd and sim dir
            for base in (Path.cwd(), SIM_DIR):
                cand = base / xp
                if cand.exists():
                    return cand.resolve()
        if xp.exists():
            return xp.resolve()
        raise FileNotFoundError(f"--xml not found: {explicit}")

    name = dataset_root.name.lower()
    seg = 2 if "2m" in name else 3 if "3m" in name else 5 if "5m" in name else None
    base = "no_base" if "no_base" in name else "with_base" if "with_base" in name else None
    if seg is None or base is None:
        raise RuntimeError(
            f"无法从数据集目录名推断 xml（dataset_root={dataset_root.name}）。请显式传 --xml tdcrX_xxx.xml"
        )

    fname = f"tdcr{seg}_{base}.xml"
    for base_dir in (dataset_root, SIM_DIR, Path.cwd()):
        cand = base_dir / fname
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        f"已推断 xml={fname}，但在 {dataset_root} / {SIM_DIR} / cwd 均未找到。请显式传 --xml。"
    )


def load_existing_motor_set(dataset_root: Path) -> tuple[set[bytes], int | None]:
    """
    读取已制作数据集中存在的 motors（用来判重，确保采样出来的新 motor 不在原数据集中）。
    优先读 motors_all.npz；否则扫描 train/val/test 下 shard-*.h5 的 'motors' 数据集。
    """
    existing: set[bytes] = set()
    dim: int | None = None

    # 1) optional merged file
    npz = dataset_root / "motors_all.npz"
    if npz.exists():
        z = np.load(str(npz))
        if "motors" not in z:
            raise ValueError(f"{npz} missing 'motors' array")
        arr = np.asarray(z["motors"], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{npz}: motors shape {arr.shape} not (N,D)")
        dim = int(arr.shape[1])
        for row in arr:
            if np.all(np.isfinite(row)):
                existing.add(row.astype(np.float32).tobytes())
        return existing, dim

    # 2) scan shards
    if h5py is None:
        raise RuntimeError("h5py 未安装：无法扫描 shard-*.h5 读取 motors。")

    for sp in ("train", "val", "test"):
        sdir = dataset_root / sp
        if not sdir.is_dir():
            continue
        for fp in sorted(sdir.glob("shard-*.h5")):
            with h5py.File(str(fp), "r") as f:
                if "motors" not in f:
                    continue
                M = np.asarray(f["motors"][:], dtype=np.float32)
                if M.ndim != 2 or M.size == 0:
                    continue
                if dim is None:
                    dim = int(M.shape[1])
                elif dim != int(M.shape[1]):
                    raise RuntimeError(f"motors dim mismatch: expect {dim}, got {M.shape[1]} in {fp}")
                for row in M:
                    # 原代码支持 NaN（缺失 motor 写 NaN）；这里判重只用全有效的样本
                    if np.all(np.isfinite(row)) and (not np.isnan(row).any()):
                        existing.add(row.astype(np.float32).tobytes())

    return existing, dim


def load_motor_norm_stats(dataset_root: Path, prefer_key: str = "all") -> tuple[np.ndarray, np.ndarray, Path | None]:
    """
    读取你 add-norm --dump_global 生成的全局 motor min/max（global_motors_*.npz）。
    找不到则用 tdcr.norm 内的函数从 shards 扫一遍（仍然“调用你已有代码”）。
    """
    preferred = dataset_root / "global_motors_scope-all.npz"
    cand = []
    if preferred.exists():
        cand.append(preferred)
    cand += sorted(dataset_root.glob("global_motors_*.npz"))

    for fp in cand:
        try:
            z = np.load(str(fp), allow_pickle=True)
        except Exception:
            continue
        if not all(k in z for k in ("keys", "mins", "maxs")):
            continue
        keys = [str(k) for k in z["keys"].tolist()]
        mins = np.asarray(z["mins"], dtype=np.float32)
        maxs = np.asarray(z["maxs"], dtype=np.float32)
        if mins.ndim != 2 or maxs.ndim != 2 or mins.shape != maxs.shape:
            continue

        idx = keys.index(prefer_key) if prefer_key in keys else 0
        return mins[idx], maxs[idx], fp

    # fallback: compute from shards (same logic as norm.py)
    if h5py is None:
        raise RuntimeError("h5py 未安装：无法从 shards 计算 motor min/max。")
    shards = tdcr_norm._list_shards(dataset_root, ["train", "val", "test"])
    mm_map = tdcr_norm._compute_global_motor_minmax(shards, scope="all")
    if not mm_map:
        raise RuntimeError("未在 shard-*.h5 中发现 motors 数据集，无法得到归一化参数。")
    mmin, mmax = mm_map.get("all", next(iter(mm_map.values())))
    return mmin.astype(np.float32), mmax.astype(np.float32), None


def quantize_ctrl(ctrl: np.ndarray, lo: np.ndarray, unique_tol: float | None) -> np.ndarray:
    """
    与 collect.py 中 continuous + unique_tol 的“量化”思路一致：
    q = round((v - lo) / tol) -> lo + q * tol
    """
    ctrl = ctrl.astype(np.float32, copy=False)
    if unique_tol is None:
        return ctrl
    tol = float(unique_tol)
    q = np.round((ctrl - lo) / tol).astype(np.int64)
    out = lo + q.astype(np.float32) * np.float32(tol)
    return out.astype(np.float32)


def sample_new_ctrls(
    n: int,
    lo: np.ndarray,
    hi: np.ndarray,
    existing: set[bytes],
    unique_tol: float | None,
    seed: int,
    max_tries: int = 200000,
) -> np.ndarray:
    """
    采样 n 个不在 existing 集合里的 motor ctrl。
    """
    rng = np.random.default_rng(seed)
    span = (hi - lo).astype(np.float32)
    out: list[np.ndarray] = []
    out_keys: set[bytes] = set()

    for _ in range(max_tries):
        ctrl = lo + span * rng.random(lo.shape, dtype=np.float32)
        ctrl = quantize_ctrl(ctrl, lo, unique_tol)
        ctrl = np.clip(ctrl, lo, hi).astype(np.float32)
        key = ctrl.tobytes()
        if key in existing or key in out_keys:
            continue
        out.append(ctrl.copy())
        out_keys.add(key)
        if len(out) >= n:
            return np.stack(out, axis=0)

    raise RuntimeError(
        f"采样失败：{max_tries} 次尝试仍没凑齐 {n} 个新 motor。试试换 --seed 或放宽 --unique_tol。"
    )


def normalize_ctrl(ctrl_raw: np.ndarray, mmin: np.ndarray, mmax: np.ndarray) -> np.ndarray:
    """
    与 norm.py 写 motors_norm 的公式一致：(x-min)/(max-min)，scale<1e-6 时置 1.0 保底。
    """
    x = np.asarray(ctrl_raw, dtype=np.float32).reshape(-1)
    mn = np.asarray(mmin, dtype=np.float32).reshape(-1)
    mx = np.asarray(mmax, dtype=np.float32).reshape(-1)
    if x.size != mn.size:
        raise ValueError(f"motor dim mismatch: ctrl={x.size}, stats={mn.size}")
    scale = (mx - mn).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return ((x - mn) / scale).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="从已有数据集读取 motors + 归一化参数，生成 3 组未包含在数据集里的 motor，并仿真渲染点云到 ./test/"
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="已制作好的数据集根目录(含 train/val/test/shard-*.h5) 或其中任意 shard-*.h5 路径",
    )
    ap.add_argument("--xml", type=str, default=None, help="MuJoCo XML 路径(不填则尝试从 dataset_root 名称推断)")
    ap.add_argument("--out_dir", type=Path, default=Path("test"), help="输出目录，默认 ./test")
    ap.add_argument("--nsample", type=int, default=3, help="生成多少组新 motor + 点云（默认 3）")
    ap.add_argument("--seed", type=int, default=2026, help="采样新 motor 的随机种子")
    ap.add_argument(
        "--unique_tol",
        type=float,
        default=1e-6,
        help="用于判重的量化粒度（与 collect 的 --unique_tol 对齐）。<=0 表示不量化。",
    )
    ap.add_argument(
        "--sample_from",
        choices=["dataset", "ctrlrange"],
        default="dataset",
        help="motor 采样边界：dataset=用数据集 min/max（归一化更稳在[0,1]）；ctrlrange=用 XML actuator_ctrlrange",
    )

    # render options (尽量和 collect 默认保持一致)
    ap.add_argument("--res_w", type=int, default=1280)
    ap.add_argument("--res_h", type=int, default=720)
    ap.add_argument("--depth_max", type=float, default=0.6)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--backend", choices=["auto", "egl", "osmesa", "glfw"], default="auto")
    ap.add_argument("--no_egl_quiet", action="store_true", help="关闭 egl quiet（默认开启）")
    ap.add_argument("--relax_max_steps", type=int, default=10000)
    ap.add_argument("--stable_vel_eps", type=float, default=2e-3)
    ap.add_argument("--stable_qpos_eps", type=float, default=1e-6)
    ap.add_argument("--stable_win", type=int, default=10)
    ap.add_argument("--no_zero_vel_each_ctrl", action="store_true", help="不在每个 ctrl 前清零速度（默认清零）")
    ap.add_argument("--sim_steps", type=int, default=50000, help="不做 relax 时固定步数（默认 50）")
    ap.add_argument("--rgb_cams", nargs="*", default=None, help="可选：指定相机 name/id；默认全相机")

    args = ap.parse_args()

    dataset_root = find_dataset_root(args.dataset)
    xml_path = infer_xml(dataset_root, args.xml)

    print_color(f"[test-gen] dataset_root = {dataset_root}")
    print_color(f"[test-gen] xml          = {xml_path}")

    existing, dim_existing = load_existing_motor_set(dataset_root)
    mmin, mmax, stats_fp = load_motor_norm_stats(dataset_root, prefer_key="all")
    if stats_fp is not None:
        print_color(f"[test-gen] motor norm stats loaded: {stats_fp.name}")
    else:
        print_color("[test-gen] motor norm stats computed from shards (no global_motors_*.npz found)")

    # --- mujoco / renderer availability check ---
    if tdcr_collect.mujoco is None or tdcr_collect.Renderer is None:
        raise RuntimeError("需要 mujoco 与 mujoco.renderer.Renderer。")
    if tdcr_collect.o3d is None:
        raise RuntimeError("需要 open3d（用于写 .ply 点云）。")

    # --- build model ---
    model = tdcr_collect.mujoco.MjModel.from_xml_path(str(xml_path))
    data = tdcr_collect.mujoco.MjData(model)
    tdcr_collect.mujoco.mj_forward(model, data)

    motor_ids, motor_names, D = tdcr_collect._detect_actuators(model)
    if dim_existing is not None and dim_existing != D:
        raise RuntimeError(f"数据集 motors 维度={dim_existing}，但 XML actuator 维度={D}，请检查是否同一配置。")
    if mmin.size != D or mmax.size != D:
        raise RuntimeError(f"归一化参数维度不匹配：stats={mmin.size}，xml motors={D}")

    # sampling bounds
    ctrl_lo = model.actuator_ctrlrange[motor_ids, 0].astype(np.float32)
    ctrl_hi = model.actuator_ctrlrange[motor_ids, 1].astype(np.float32)

    if args.sample_from == "dataset":
        samp_lo = np.maximum(ctrl_lo, mmin.astype(np.float32))
        samp_hi = np.minimum(ctrl_hi, mmax.astype(np.float32))
        bad = samp_hi <= samp_lo
        if np.any(bad):
            print_color("[test-gen] WARNING: dataset bounds invalid on some dims; fallback those dims to XML ctrlrange.")
            samp_lo[bad] = ctrl_lo[bad]
            samp_hi[bad] = ctrl_hi[bad]
    else:
        samp_lo, samp_hi = ctrl_lo, ctrl_hi

    unique_tol = None if (args.unique_tol is None or args.unique_tol <= 0) else float(args.unique_tol)

    ctrls_raw = sample_new_ctrls(args.nsample, samp_lo, samp_hi, existing, unique_tol, args.seed)

    # output dirs
    out_dir = ensure_dir(args.out_dir)
    out_pcd = ensure_dir(out_dir / "pointcloud")
    out_motor = ensure_dir(out_dir / "motor")

    # renderer
    renderer = tdcr_collect._make_renderer_robust(
        model, args.res_w, args.res_h, prefer=args.backend, egl_quiet=(not args.no_egl_quiet)
    )

    # camera selection (same as collect.py)
    _, all_cam_names = tdcr_collect._detect_cameras(model)
    cam_ids = tdcr_collect._select_camera_ids(args.rgb_cams, all_cam_names)

    # pixel subsampling index
    U_full, V_full = np.meshgrid(
        np.arange(args.res_w, dtype=np.float32),
        np.arange(args.res_h, dtype=np.float32),
    )
    U_s = U_full[:: args.stride, :: args.stride].ravel().astype(np.intp)
    V_s = V_full[:: args.stride, :: args.stride].ravel().astype(np.intp)
    idx_s = (V_s, U_s)

    zero_vel = not bool(args.no_zero_vel_each_ctrl)

    # main loop
    for i, ctrl_raw in enumerate(ctrls_raw, start=1):
        stem = str(i).zfill(6)

        # set ctrl and relax
        data.ctrl[motor_ids] = ctrl_raw.astype(np.float32)
        if args.relax_max_steps and args.stable_win:
            tdcr_collect._relax_to_stable(
                model,
                data,
                max_steps=args.relax_max_steps,
                vel_eps=args.stable_vel_eps,
                qpos_eps=args.stable_qpos_eps,
                win=args.stable_win,
                zero_vel=zero_vel,
            )
        else:
            for _ in range(int(args.sim_steps)):
                tdcr_collect.mujoco.mj_step(model, data)

        # render point cloud from cameras
        pts_all, col_all = [], []
        for cid in cam_ids:
            rgb, depth = tdcr_collect._render_rgbd(renderer, data, cid)
            fx, fy, cx, cy = tdcr_collect._intrinsics(model, args.res_w, args.res_h, cid)
            R = data.cam_xmat[cid].reshape(3, 3).astype(np.float32)
            p = data.cam_xpos[cid].astype(np.float32)
            pts, col = tdcr_collect._cam_to_world_fast(
                rgb, depth, fx, fy, cx, cy, R, p, args.depth_max, idx_s
            )
            if pts is not None:
                pts_all.append(pts)
                col_all.append(col)

        if not pts_all:
            print_color(f"[test-gen] WARNING: {stem}: empty pointcloud (try larger --depth_max?)")
            pts = np.zeros((0, 3), np.float32)
            col = np.zeros((0, 3), np.float32)
        else:
            pts = np.concatenate(pts_all, axis=0).astype(np.float32)
            col = np.concatenate(col_all, axis=0).astype(np.float32)

        # save ply
        pcd_path = out_pcd / f"{stem}.ply"
        pcd = tdcr_collect.o3d.geometry.PointCloud()
        pcd.points = tdcr_collect.o3d.utility.Vector3dVector(pts)
        pcd.colors = tdcr_collect.o3d.utility.Vector3dVector(col)
        tdcr_collect.o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)

        # normalize motor and save json
        ctrl_norm = normalize_ctrl(ctrl_raw, mmin, mmax)

        # ✅ 这里把 "ctrl" 设置为归一化后的（满足“电机数据要归一化”的需求）
        # 同时保留 ctrl_raw，方便你追溯/复现实验
        motor_obj = {
            "ctrl": [float(x) for x in ctrl_norm.tolist()],         # normalized (primary)
            "ctrl_raw": [float(x) for x in ctrl_raw.tolist()],      # raw used in sim
            "actuator_names": motor_names,
            "norm": {
                "type": "minmax",
                "key": "all",
                "stats_file": (str(stats_fp) if stats_fp is not None else None),
                "min": [float(x) for x in mmin.tolist()],
                "max": [float(x) for x in mmax.tolist()],
            },
        }
        motor_path = out_motor / f"{stem}.json"
        with open(motor_path, "w", encoding="utf-8") as f:
            json.dump(motor_obj, f, indent=2, ensure_ascii=False)

        print_color(f"[test-gen] wrote {pcd_path} + {motor_path}")

    renderer.close()
    print_color("✅ [test-gen] done.")


if __name__ == "__main__":
    main()
'''
export MUJOCO_GL=egl
export EGL_LOG_LEVEL=fatal
export LIBEGL_DEBUG=fatal

python make_test_from_dataset.py --dataset /data/yxk/K-data/K/fllm-sm/datasets/sim/2m_no_base --out_dir test/2m_no_base

python make_test_from_dataset.py --dataset 2m_with_base --out_dir test/2m_with_base

python make_test_from_dataset.py --dataset 3m_no_base --out_dir test/3m_no_base --depth_max 2.0

python make_test_from_dataset.py --dataset 3m_with_base --out_dir test/3m_with_base --depth_max 2.0

python make_test_from_dataset.py --dataset 5m_no_base --out_dir test/5m_no_base --depth_max 2.0

python make_test_from_dataset.py --dataset 5m_with_base --out_dir test/5m_with_base --depth_max 2.0


'''