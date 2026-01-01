from __future__ import annotations

import os
import json
import glob
import argparse
import warnings
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import h5py
from tqdm import tqdm

# Your model definitions (must include VelocityNet and HybridMLP)
from models import VelocityNet
try:
    from models import HybridMLP
except Exception:
    HybridMLP = None


# ============================================================
# Chamfer Distance: prefer compiled chamfer_3D if available
# ============================================================
_CHAMFER_EXT = None
_CHAMFER_EXT_FAILED = False


def _load_chamfer_ext():
    global _CHAMFER_EXT, _CHAMFER_EXT_FAILED
    if _CHAMFER_EXT is not None or _CHAMFER_EXT_FAILED:
        return _CHAMFER_EXT
    try:
        import importlib
        _CHAMFER_EXT = importlib.import_module("chamfer_3D")
        print("[Chamfer] Using compiled chamfer_3D extension.")
    except Exception as e:
        warnings.warn(
            f"[Chamfer] chamfer_3D not available: {e}. Falling back to torch.cdist.",
            RuntimeWarning,
        )
        _CHAMFER_EXT_FAILED = True
        _CHAMFER_EXT = None
    return _CHAMFER_EXT


@torch.no_grad()
def chamfer_l2(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> torch.Tensor:
    """
    pred_xyz, gt_xyz: (B, N, 3)
    return: (B,) squared-L2 Chamfer
    """
    assert pred_xyz.shape[-1] == 3 and gt_xyz.shape[-1] == 3, "Chamfer expects xyz only."
    ext = _load_chamfer_ext()
    if ext is not None and pred_xyz.is_cuda and gt_xyz.is_cuda:
        B, N, _ = pred_xyz.shape
        x = pred_xyz.contiguous().to(dtype=torch.float32)
        y = gt_xyz.contiguous().to(dtype=torch.float32)

        d1 = torch.empty(B, N, device=x.device, dtype=torch.float32)
        d2 = torch.empty(B, N, device=x.device, dtype=torch.float32)
        i1 = torch.empty(B, N, device=x.device, dtype=torch.int32)
        i2 = torch.empty(B, N, device=x.device, dtype=torch.int32)

        _ = ext.forward(x, y, d1, d2, i1, i2)
        return (d1.mean(dim=1) + d2.mean(dim=1)).to(pred_xyz.dtype)

    # fallback
    d2 = torch.cdist(pred_xyz, gt_xyz, p=2).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


# ============================================================
# Simple PLY writers (ASCII)
# ============================================================
def write_ply_xyz(path: str, points_xyz: np.ndarray):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, points_xyz, fmt="%.6f %.6f %.6f")


def write_ply_xyzrgb(path: str, points_xyzrgb: np.ndarray):
    """
    points_xyzrgb: (N,6) with rgb in [0,1]
    """
    points_xyzrgb = np.asarray(points_xyzrgb, dtype=np.float32)
    assert points_xyzrgb.ndim == 2 and points_xyzrgb.shape[1] == 6
    xyz = points_xyzrgb[:, :3]
    rgb = points_xyzrgb[:, 3:6]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb255 = (rgb * 255.0).clip(0.0, 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb255):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


# ============================================================
# Samplers
# ============================================================
@torch.no_grad()
def euler_sampler(
    net: torch.nn.Module,
    x0: torch.Tensor,
    cond: Optional[torch.Tensor],
    steps: int,
    guidance_scale: float,
    clamp_rgb: bool = True,
) -> torch.Tensor:
    """Euler sampler for dx/dt = v(x,t,cond), integrate t:0->1"""
    dt = 1.0 / float(steps)
    x = x0
    for i in range(steps):
        t = torch.full((x.shape[0],), (i + 0.5) * dt, device=x.device, dtype=x.dtype)
        v = net.guided_velocity(x, t, cond, guidance_scale=guidance_scale)
        x = x + v * dt
        if clamp_rgb and x.shape[-1] == 6:
            x[..., 3:] = x[..., 3:].clamp(0.0, 1.0)
    return x


@torch.no_grad()
def heun_sampler(
    net: torch.nn.Module,
    x0: torch.Tensor,
    cond: Optional[torch.Tensor],
    steps: int,
    guidance_scale: float,
    clamp_rgb: bool = True,
) -> torch.Tensor:
    """Heun / RK2 sampler for dx/dt = v(x,t,cond), integrate t:0->1"""
    dt = 1.0 / float(steps)
    x = x0
    B = x.shape[0]
    for i in range(steps):
        t0 = torch.full((B,), i * dt, device=x.device, dtype=x.dtype)
        v1 = net.guided_velocity(x, t0, cond, guidance_scale=guidance_scale)
        x_euler = x + v1 * dt

        t1 = torch.full((B,), (i + 1) * dt, device=x.device, dtype=x.dtype)
        v2 = net.guided_velocity(x_euler, t1, cond, guidance_scale=guidance_scale)

        x = x + 0.5 * dt * (v1 + v2)

        if clamp_rgb and x.shape[-1] == 6:
            x[..., 3:] = x[..., 3:].clamp(0.0, 1.0)
    return x


# ============================================================
# H5 utils
# ============================================================
def find_h5_files(data_dir: str, split: str) -> List[str]:
    patterns = [
        os.path.join(data_dir, split, "*.h5"),
        os.path.join(data_dir, split, "*.hdf5"),
        os.path.join(data_dir, f"{split}*.h5"),
        os.path.join(data_dir, f"{split}*.hdf5"),
        os.path.join(data_dir, "*.h5"),
        os.path.join(data_dir, "*.hdf5"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files))


def _pick_points_key(f: h5py.File, prefer_norm: bool) -> str:
    if prefer_norm and ("data_norm" in f):
        return "data_norm"
    if "data" in f:
        return "data"
    if "data_norm" in f:
        return "data_norm"
    raise KeyError(f"H5 missing points key. Need data_norm/data. keys={list(f.keys())}")


def _pick_cond_key(f: h5py.File) -> str:
    # compat: motors_norm / motor_norm / motors
    if "motors_norm" in f:
        return "motors_norm"
    if "motor_norm" in f:
        return "motor_norm"
    if "motors" in f:
        return "motors"
    raise KeyError(
        f"H5 missing motor key. Need motors_norm/motor_norm/motors. keys={list(f.keys())}"
    )


def _pick_rgb_key(f: h5py.File, rgb_key: str) -> str:
    if rgb_key in f:
        return rgb_key
    if "rgb" in f:
        return "rgb"
    raise KeyError(f"H5 missing rgb key. want={rgb_key}, keys={list(f.keys())}")


def rgb_to_01(rgb: np.ndarray) -> np.ndarray:
    """
    Accept rgb in:
      - uint8 [0,255]
      - float [0,1]
      - (optional) float [-1,1] (legacy) -> map to [0,1]
    Return float32 in [0,1].
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size == 0:
        return rgb
    mx = float(np.max(rgb))
    mn = float(np.min(rgb))
    if mx > 1.0:
        rgb = rgb / 255.0
    elif mn < 0.0:
        # legacy [-1,1] -> [0,1]
        rgb = (rgb + 1.0) * 0.5
    return np.clip(rgb, 0.0, 1.0)


# ============================================================
# Subsampling
# ============================================================
def subsample_np(points: np.ndarray, k: Optional[int], rng: np.random.RandomState) -> np.ndarray:
    if k is None or k <= 0:
        return points
    n = int(points.shape[0])
    if k >= n:
        return points
    idx = rng.choice(n, size=k, replace=False)
    return points[idx]


def subsample_torch_per_example(
    x: torch.Tensor, k: Optional[int], rng: np.random.RandomState
) -> torch.Tensor:
    """
    x: (B,N,C) -> (B,k,C) using numpy rng for max compatibility
    """
    if k is None or k <= 0:
        return x
    B, N, C = x.shape
    if k >= N:
        return x
    out = []
    for b in range(B):
        idx_np = rng.choice(N, size=k, replace=False).astype(np.int64)
        idx = torch.from_numpy(idx_np).to(device=x.device, dtype=torch.long)
        out.append(x[b, idx, :])
    return torch.stack(out, dim=0)


# ============================================================
# Model backbone inference & builder
# ============================================================
def infer_backbone(ckpt: Dict[str, Any]) -> str:
    args = ckpt.get("args", {}) or {}
    pf = args.get("pf_backbone", None)
    if pf in ("mlp", "hybrid"):
        return pf

    sd = ckpt.get("model", {}) or {}
    keys = list(sd.keys())
    if any(k.startswith("ctx_net.") for k in keys) or any(k.startswith("head.") for k in keys):
        return "hybrid"
    return "mlp"


def _to_int_list(x, default: List[int]) -> List[int]:
    if x is None:
        return list(default)
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    # single value
    return [int(x)]


def build_model_from_ckpt_args(backbone: str, ckpt_args: Dict[str, Any]) -> torch.nn.Module:
    cond_dim = int(ckpt_args.get("cond_dim", 0))
    width = int(ckpt_args.get("width", 512))
    depth = int(ckpt_args.get("depth", 6))
    emb_dim = int(ckpt_args.get("emb_dim", 256))
    cfg_drop_p = float(ckpt_args.get("cfg_drop_p", 0.0))

    use_rgb = bool(ckpt_args.get("use_rgb", False))
    point_dim = int(ckpt_args.get("point_dim", 6 if use_rgb else 3))

    if backbone == "mlp":
        return VelocityNet(
            cond_dim=cond_dim,
            point_dim=point_dim,
            width=width,
            depth=depth,
            emb_dim=emb_dim,
            cfg_dropout_p=cfg_drop_p,
        )

    if HybridMLP is None:
        raise ImportError("Checkpoint seems hybrid, but HybridMLP is not available in current project.")

    return HybridMLP(
        cond_dim=cond_dim,
        point_dim=point_dim,
        ctx_dim=int(ckpt_args.get("ctx_dim", 64)),
        ctx_emb_dim=int(ckpt_args.get("ctx_emb_dim", emb_dim)),
        stage_channels=_to_int_list(ckpt_args.get("ctx_stage_channels", None), [128, 256, 256]),
        stage_blocks=_to_int_list(ckpt_args.get("ctx_stage_blocks", None), [2, 2, 2]),
        stage_res=_to_int_list(ckpt_args.get("ctx_stage_res", None), [32, 16, 8]),
        with_se=bool(ckpt_args.get("ctx_with_se", True)),
        norm_type=str(ckpt_args.get("ctx_norm", "group")),
        gn_groups=int(ckpt_args.get("ctx_gn_groups", 32)),
        with_global=bool(ckpt_args.get("ctx_with_global", True)),
        voxel_normalize=bool(ckpt_args.get("ctx_voxel_normalize", True)),
        use_t_gate=True,
        t_gate_k=float(ckpt_args.get("ctx_t_gate_k", 10.0)),
        t_gate_tau=float(ckpt_args.get("ctx_t_gate_tau", 0.8)),
        pf_width=width,
        pf_depth=depth,
        pf_emb_dim=emb_dim,
        cfg_dropout_p=cfg_drop_p,
    )


# ============================================================
# Prior (match training)
# ============================================================
def make_prior_like(
    gt: torch.Tensor,
    prior_std: float,
    color_prior: str = "uniform",
    color_prior_std: float = 1.0,
) -> torch.Tensor:
    """
    gt: (B,N,3) or (B,N,6)
    xyz: N(0, prior_std^2)
    rgb (6D): by color_prior, default uniform [0,1]
    """
    if gt.shape[-1] == 3:
        return torch.randn_like(gt) * float(prior_std)

    if gt.shape[-1] != 6:
        raise ValueError(f"Unsupported point_dim in prior: {gt.shape[-1]}")

    z = torch.empty_like(gt)
    z[..., :3] = torch.randn_like(gt[..., :3]) * float(prior_std)

    cp = str(color_prior)
    if cp == "uniform":
        z[..., 3:] = torch.rand_like(gt[..., 3:])
    elif cp == "zeros":
        z[..., 3:] = 0.0
    elif cp == "gauss":
        z[..., 3:] = torch.randn_like(gt[..., 3:]) * float(color_prior_std)
    else:
        raise ValueError(f"Unknown color_prior: {cp}")
    return z


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("TDCR demo generation (supports xyz / xyzrgb)")

    ap.add_argument("--ckpt", type=str, required=True, help="path to latest.pt / epoch_xxxx.pt")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--demo_out", type=str, required=True)

    ap.add_argument("--use_norm", action="store_true", default=True, help="prefer data_norm if exists")
    ap.add_argument("--max_points", type=int, default=0,
                    help="Max points for generation/saving (<=0: use ckpt te_max_sample_points; still 0 -> full)")
    ap.add_argument("--cd_points", type=int, default=0,
                    help="Points for CD computation (<=0: use max_points/full)")

    ap.add_argument("--sample_steps", type=int, default=0, help="Override ckpt sample_steps (<=0: no override)")
    ap.add_argument("--prior_std", type=float, default=0.0, help="Override ckpt prior_std / point_prior_std (<=0: no override)")
    ap.add_argument("--guidance_scale", type=float, default=None, help="Override ckpt guidance_scale (if set)")

    ap.add_argument("--color_prior", type=str, default=None, choices=["uniform", "zeros", "gauss"],
                    help="Override ckpt color_prior for 6D (default: use ckpt)")
    ap.add_argument("--color_prior_std", type=float, default=None,
                    help="Override ckpt color_prior_std (only for gauss prior)")

    ap.add_argument("--rgb_key", type=str, default=None,
                    help="Override ckpt rgb_key (only used when point_dim==6)")

    ap.add_argument("--sampler", type=str, default="heun", choices=["heun", "euler"],
                    help="Sampling ODE solver (default: heun/RK2).")
    ap.add_argument("--no_clamp_rgb", action="store_true", default=False,
                    help="Disable clamping rgb to [0,1] during sampling (not recommended).")

    ap.add_argument("--eval_fraction", type=float, default=1.0,
                    help="Evaluate only this fraction of split samples (0<..<=1). e.g. 0.1 for 10%.")
    ap.add_argument("--eval_max_samples", type=int, default=0,
                    help="Cap number of evaluated samples after applying eval_fraction (0: no cap).")
    ap.add_argument("--eval_seed", type=int, default=None,
                    help="Seed used to choose subset samples (default: --seed).")
    ap.add_argument("--no_save_ply", action="store_true", default=False,
                    help="If set, do not write gt/pred ply files (faster).")
    ap.add_argument("--no_ema", action="store_true", default=False, help="Do not use EMA weights")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # seed (global) + numpy rng
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np_rng = np.random.RandomState(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}

    backbone = infer_backbone(ckpt)
    print(f"[Demo] Backbone inferred: {backbone}")

    # infer point_dim & rgb_key from ckpt
    use_rgb = bool(ckpt_args.get("use_rgb", False))
    point_dim = int(ckpt_args.get("point_dim", 6 if use_rgb else 3))
    if point_dim not in (3, 6):
        raise ValueError(f"[Demo] Unsupported point_dim={point_dim} (expect 3 or 6).")

    # sampler hyperparams: default from ckpt, allow override
    sample_steps = int(ckpt_args.get("sample_steps", 50))
    if args.sample_steps and args.sample_steps > 0:
        sample_steps = int(args.sample_steps)

    # prior_std: prefer ckpt prior_std, fallback to point_prior_std (legacy naming)
    prior_std = ckpt_args.get("prior_std", None)
    if prior_std is None:
        prior_std = ckpt_args.get("point_prior_std", 1.0)
    prior_std = float(prior_std)
    if args.prior_std and args.prior_std > 0:
        prior_std = float(args.prior_std)

    guidance_scale = float(ckpt_args.get("guidance_scale", 0.0))
    if args.guidance_scale is not None:
        guidance_scale = float(args.guidance_scale)

    # 6D prior knobs
    color_prior = str(ckpt_args.get("color_prior", "uniform"))
    color_prior_std = float(ckpt_args.get("color_prior_std", 1.0))
    if args.color_prior is not None:
        color_prior = str(args.color_prior)
    if args.color_prior_std is not None:
        color_prior_std = float(args.color_prior_std)

    rgb_key = str(ckpt_args.get("rgb_key", "rgb"))
    if args.rgb_key is not None:
        rgb_key = str(args.rgb_key)

    # max_points: default to ckpt te_max_sample_points
    max_points: Optional[int] = None
    if args.max_points and args.max_points > 0:
        max_points = int(args.max_points)
    else:
        te_mp = ckpt_args.get("te_max_sample_points", None)
        if te_mp is not None:
            te_mp = int(te_mp)
            if te_mp > 0:
                max_points = te_mp

    cd_points: Optional[int] = int(args.cd_points) if args.cd_points and args.cd_points > 0 else None
    if cd_points is None:
        cd_points = max_points  # if still None -> later use full

    # build model
    net = build_model_from_ckpt_args(backbone, ckpt_args).to(device)
    net.eval()

    # load weights (prefer EMA unless --no_ema)
    if (not args.no_ema) and (ckpt.get("ema", None) is not None):
        state = ckpt["ema"]
        print("[Demo] Loading EMA weights.")
    else:
        state = ckpt.get("model", None)
        print("[Demo] Loading model weights.")
    if state is None:
        raise KeyError("Checkpoint missing both 'ema' and 'model'.")

    # strict load if possible
    try:
        net.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[Demo][WARN] strict load failed: {e}. Retrying with strict=False.")
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print(f"[Demo][WARN] Missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[Demo][WARN] Unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    # output dirs
    out_gt = os.path.join(args.demo_out, "gt")
    out_pred = os.path.join(args.demo_out, "pred")
    if not args.no_save_ply:
        os.makedirs(out_gt, exist_ok=True)
        os.makedirs(out_pred, exist_ok=True)

        # scan split files
        files = find_h5_files(args.data_dir, args.split)
        if not files:
            raise FileNotFoundError(f"No .h5/.hdf5 found. data_dir={args.data_dir} split={args.split}")

        # choose subset of samples to evaluate (optional)
        eval_fraction = float(args.eval_fraction)
        if not (0.0 < eval_fraction <= 1.0):
            raise ValueError(f"--eval_fraction must be in (0,1], got {eval_fraction}")
        eval_seed = int(args.eval_seed) if args.eval_seed is not None else int(args.seed)
        subset_rng = np.random.RandomState(eval_seed)

        # Count how many examples exist in this split (across all shards),
        # then pick a random subset without replacement.
        import bisect
        file_counts: List[int] = []
        cum_counts: List[int] = []
        total = 0
        for _fp in files:
            with h5py.File(_fp, "r") as _f:
                _pts_key = _pick_points_key(_f, prefer_norm=args.use_norm)
                _B = int(_f[_pts_key].shape[0])
            file_counts.append(_B)
            total += _B
            cum_counts.append(total)

        if total <= 0:
            raise RuntimeError(f"No samples found in split='{args.split}' under {args.data_dir}")

        k = total
        if eval_fraction < 1.0:
            k = max(1, int(round(total * eval_fraction)))
        if args.eval_max_samples and args.eval_max_samples > 0:
            k = min(k, int(args.eval_max_samples))

        if k < total:
            sel = subset_rng.choice(total, size=k, replace=False)
            sel = np.sort(sel).tolist()
        else:
            sel = list(range(total))

        # Map selected global ids -> (file, local_idx) for efficient H5 access
        selected_by_file: Dict[str, List[tuple[int, int]]] = {}
        for gid in sel:
            fi = bisect.bisect_right(cum_counts, gid)
            start_off = cum_counts[fi - 1] if fi > 0 else 0
            local_i = int(gid - start_off)
            selected_by_file.setdefault(files[fi], []).append((int(gid), local_i))

        # Sort within each file so H5 reads are more sequential
        for _fp, items in selected_by_file.items():
            items.sort(key=lambda x: x[1])

        print(f"[Demo] Eval subset: {k}/{total} samples ({100.0 * k / total:.2f}%), seed={eval_seed}")

        clamp_rgb = not args.no_clamp_rgb
        sampler_fn = heun_sampler if args.sampler == "heun" else euler_sampler

        print(
            f"[Demo] Found {len(files)} h5 shards. device={device} point_dim={point_dim} steps={sample_steps} "
            f"prior_std={prior_std} max_points={max_points} cd_points={cd_points} guidance_scale={guidance_scale} "
            f"sampler={args.sampler} clamp_rgb={clamp_rgb} color_prior={color_prior}"
        )

        cds: List[float] = []
        sample_idx = 0

        # Iterate shards, but only process selected indices in each shard
        for fp in files:
            items = selected_by_file.get(fp, None)
            if not items:
                continue

            with h5py.File(fp, "r") as f:
                pts_key = _pick_points_key(f, prefer_norm=args.use_norm)
                pts_ds = f[pts_key]      # (B,N,3) or (B,N,6)

                # cond may not exist for unconditional models
                cond_dim = int(ckpt_args.get("cond_dim", 0))
                cond_ds = None
                if cond_dim > 0:
                    cond_key = _pick_cond_key(f)
                    cond_ds = f[cond_key]  # (B,D)

                rgb_ds = None
                if point_dim == 6 and pts_ds.shape[-1] == 3:
                    rgb_ds = f[_pick_rgb_key(f, rgb_key)]  # (B,N,3)

                for _gid, i in tqdm(items, desc=f"[{os.path.basename(fp)}]", leave=False):
                    sample_idx += 1
                    name = f"{sample_idx:06d}.ply"

                    # load gt xyz (or xyzrgb)
                    gt_np = np.asarray(pts_ds[i], dtype=np.float32)
                    if gt_np.ndim != 2:
                        raise ValueError(f"Unexpected points shape: {gt_np.shape} in {fp}")

                    if point_dim == 6:
                        if gt_np.shape[1] == 6:
                            # already concatenated; normalize rgb just in case
                            gt_np[:, 3:6] = rgb_to_01(gt_np[:, 3:6])
                        else:
                            if rgb_ds is None:
                                raise KeyError(f"[Demo] point_dim=6 but rgb not found in H5: {fp}")
                            rgb_np = rgb_to_01(np.asarray(rgb_ds[i]))
                            if rgb_np.shape[0] != gt_np.shape[0]:
                                raise ValueError(f"xyz/rgb length mismatch: xyz={gt_np.shape} rgb={rgb_np.shape}")
                            gt_np = np.concatenate([gt_np, rgb_np.astype(np.float32)], axis=1)

                    # optional subsample for generation/saving
                    if max_points is not None:
                        gt_np = subsample_np(gt_np, max_points, np_rng)

                    # save gt
                    if not args.no_save_ply:
                        if point_dim == 6:
                            write_ply_xyzrgb(os.path.join(out_gt, name), gt_np)
                        else:
                            write_ply_xyz(os.path.join(out_gt, name), gt_np[:, :3])

                    # cond
                    cond = None
                    if cond_ds is not None:
                        cond_np = np.asarray(cond_ds[i], dtype=np.float32).reshape(-1)
                        cond = torch.from_numpy(cond_np)[None, ...].to(device=device, dtype=torch.float32)

                    # to torch
                    gt = torch.from_numpy(gt_np)[None, ...].to(device=device, dtype=torch.float32)

                    # sample
                    z = make_prior_like(gt, prior_std, color_prior=color_prior, color_prior_std=color_prior_std)
                    pred = sampler_fn(net, z, cond, steps=sample_steps, guidance_scale=guidance_scale, clamp_rgb=clamp_rgb)

                    # save pred
                    if not args.no_save_ply:
                        pred_np = pred[0].detach().cpu().numpy()
                        if point_dim == 6:
                            pred_np[:, 3:6] = np.clip(pred_np[:, 3:6], 0.0, 1.0)
                            write_ply_xyzrgb(os.path.join(out_pred, name), pred_np)
                        else:
                            write_ply_xyz(os.path.join(out_pred, name), pred_np[:, :3])

                    # CD (xyz only)
                    pred_cd = pred[..., :3]
                    gt_cd = gt[..., :3]

                    if cd_points is not None and cd_points > 0:
                        pred_cd = subsample_torch_per_example(pred_cd, cd_points, np_rng)
                        gt_cd = subsample_torch_per_example(gt_cd, cd_points, np_rng)

                    cd = float(chamfer_l2(pred_cd, gt_cd)[0].item())
                    cds.append(cd)

        mean_cd = float(np.mean(cds)) if cds else float("nan")
        std_cd = float(np.std(cds)) if cds else float("nan")
        print(f"[Demo] Done. Samples={len(cds)} mean_CD={mean_cd:.8f} std_CD={std_cd:.8f}")

        # write summary
        os.makedirs(args.demo_out, exist_ok=True)
        with open(os.path.join(args.demo_out, "summary.json"), "w") as f:
            json.dump(
                {
                    "ckpt": args.ckpt,
                    "data_dir": args.data_dir,
                    "split": args.split,
                    "backbone": backbone,
                    "point_dim": point_dim,
                    "samples": len(cds),
                    "mean_cd": mean_cd,
                    "sample_steps": sample_steps,
                    "prior_std": prior_std,
                    "guidance_scale": guidance_scale,
                    "max_points": max_points,
                    "cd_points": cd_points,
                    "sampler": args.sampler,
                    "clamp_rgb": clamp_rgb,
                    "color_prior": color_prior,
                    "color_prior_std": color_prior_std,
                    "rgb_key": rgb_key,
                    "eval_fraction": eval_fraction,
                    "eval_max_samples": int(args.eval_max_samples),
                    "eval_seed": eval_seed,
                    "eval_total_samples": total,
                    "eval_selected_samples": k,
                    "no_save_ply": bool(args.no_save_ply),
                    "std_cd": std_cd,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()

'''
export CUDA_VISIBLE_DEVICES=5
python demo_generate_tdcr.py \
  --ckpt runs_final/sim_5m_with_base_hybrid_12_28/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out demo_out/sim_5m_with_base_hybrid_12_28_eval10p \
  --sampler heun \
  --sample_steps 50 \
  --prior_std 0.5 \
  --rgb_key rgb \
  --cd_points 4096 \
  --eval_fraction 0.1 \
  --eval_seed 42 \
  --no_save_ply


export CUDA_VISIBLE_DEVICES=5
python demo_generate_tdcr.py \
  --ckpt runs_final/sim_5m_with_base_mlp_12_27/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out demo_out/sim_5m_with_base_mlp \
  --sampler heun \
  --sample_steps 50 \
  --prior_std 0.5 \
  --rgb_key rgb \
  --cd_points 4096 \
  --eval_fraction 0.1 \
  --eval_seed 42 


export CUDA_VISIBLE_DEVICES=5
python demo_generate_tdcr.py \
  --ckpt runs_final/sim_3m_with_base_mlp_12_27/ckpts/latest.pt \
  --data_dir datasets/sim/3m_with_base \
  --split test \
  --demo_out demo_out/sim_3m_with_base_mlp_12_27 \
  --sampler heun \
  --sample_steps 50 \
  --prior_std 0.5 \
  --rgb_key rgb \
  --cd_points 4096 \
  --eval_fraction 0.1 \
  --eval_seed 42

export CUDA_VISIBLE_DEVICES=4
python demo_generate_tdcr.py \
  --ckpt runs_final/sim_3m_with_base_hybrid_12_28/ckpts/latest.pt \
  --data_dir datasets/sim/3m_with_base \
  --split test \
  --demo_out demo_out/sim_3m_with_base_hybrid_12_28 \
  --sampler heun \
  --sample_steps 50 \
  --prior_std 0.5 \
  --rgb_key rgb \
  --cd_points 4096 \
  --eval_fraction 0.1 \
  --eval_seed 42

export CUDA_VISIBLE_DEVICES=1
python demo_generate_tdcr.py \
  --ckpt runs_final/sim_5m_with_base_hybrid_12_30_new_hybrid_params_half_data/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out demo_out/sim_5m_with_base_hybrid_12_30_new_hybrid_params_half_data \
  --sampler heun \
  --sample_steps 100 \
  --prior_std 0.5 \
  --rgb_key rgb \
  --cd_points 4096 \
  --eval_fraction 0.1 \
  --eval_seed 42


export CUDA_VISIBLE_DEVICES=1
python demo_generate_tdcr.py \
  --ckpt /data/fllm/code/TDCR-Self-Modeling/runs_final/sim_5m_with_base_hybrid_bighead_tau0.8/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out demo_out/sim_5m_with_base_hybrid_bighead_tau0.8 \
  --sampler heun \
  --sample_steps 50 \
  --prior_std 0.5 \
  --rgb_key rgb \
  --cd_points 4096 \
  --eval_fraction 0.1 \
  --eval_seed 42
'''