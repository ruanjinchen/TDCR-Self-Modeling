from __future__ import annotations

import os
import re
import json
import glob
import argparse
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import h5py
from tqdm import tqdm

# 你的模型定义（需要同时包含 VelocityNet 和 HybridMLP）
from models import VelocityNet
try:
    from models import HybridMLP
except Exception:
    HybridMLP = None


# -------------------------
# Chamfer Distance: prefer compiled chamfer_3D
# -------------------------
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
        warnings.warn(f"[Chamfer] chamfer_3D not available: {e}. Falling back to torch.cdist.", RuntimeWarning)
        _CHAMFER_EXT_FAILED = True
        _CHAMFER_EXT = None
    return _CHAMFER_EXT

@torch.no_grad()
def chamfer_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    pred, gt: (B, N, 3)
    return: (B,) squared-L2 Chamfer
    """
    ext = _load_chamfer_ext()
    if ext is not None and pred.is_cuda and gt.is_cuda:
        B, N, _ = pred.shape
        x = pred.contiguous().to(dtype=torch.float32)
        y = gt.contiguous().to(dtype=torch.float32)

        d1 = torch.empty(B, N, device=x.device, dtype=torch.float32)
        d2 = torch.empty(B, N, device=x.device, dtype=torch.float32)
        i1 = torch.empty(B, N, device=x.device, dtype=torch.int32)
        i2 = torch.empty(B, N, device=x.device, dtype=torch.int32)

        _ = ext.forward(x, y, d1, d2, i1, i2)
        return (d1.mean(dim=1) + d2.mean(dim=1)).to(pred.dtype)

    # fallback
    d2 = torch.cdist(pred, gt, p=2).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


# -------------------------
# Simple PLY writer (ASCII)
# -------------------------
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


# -------------------------
# Euler sampler
# -------------------------
@torch.no_grad()
def euler_sampler(net: torch.nn.Module,
                  x0: torch.Tensor,
                  cond: Optional[torch.Tensor],
                  steps: int,
                  guidance_scale: float) -> torch.Tensor:
    dt = 1.0 / float(steps)
    x = x0
    for i in range(steps):
        t = torch.full((x.shape[0],), (i + 0.5) * dt, device=x.device, dtype=x.dtype)
        v = net.guided_velocity(x, t, cond, guidance_scale=guidance_scale)
        x = x + v * dt
    return x


# -------------------------
# H5 utils
# -------------------------
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
    # 兼容 motors_norm / motor_norm / motors
    if "motors_norm" in f:
        return "motors_norm"
    if "motor_norm" in f:
        return "motor_norm"
    if "motors" in f:
        return "motors"
    raise KeyError(f"H5 missing motor key. Need motors_norm/motor_norm/motors. keys={list(f.keys())}")


# -------------------------
# Subsampling
# -------------------------
def subsample_np(points: np.ndarray, k: Optional[int], rng: np.random.RandomState) -> np.ndarray:
    if k is None or k <= 0:
        return points
    n = int(points.shape[0])
    if k >= n:
        return points
    idx = rng.choice(n, size=k, replace=False)
    return points[idx]

def subsample_torch_per_example(x: torch.Tensor, k: Optional[int], rng: np.random.RandomState) -> torch.Tensor:
    """
    x: (B,N,3) -> (B,k,3) using numpy rng for max compatibility
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


# -------------------------
# Model backbone inference
# -------------------------
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

def build_model_from_ckpt_args(backbone: str, ckpt_args: Dict[str, Any]) -> torch.nn.Module:
    cond_dim = int(ckpt_args.get("cond_dim", 0))
    width = int(ckpt_args.get("width", 512))
    depth = int(ckpt_args.get("depth", 6))
    emb_dim = int(ckpt_args.get("emb_dim", 256))
    cfg_drop_p = float(ckpt_args.get("cfg_drop_p", 0.0))

    if backbone == "mlp":
        return VelocityNet(
            cond_dim=cond_dim,
            width=width,
            depth=depth,
            emb_dim=emb_dim,
            cfg_dropout_p=cfg_drop_p,
        )

    if HybridMLP is None:
        raise ImportError("Checkpoint seems hybrid, but HybridMLP is not available in current project.")

    return HybridMLP(
        cond_dim=cond_dim,
        point_dim=3,
        ctx_dim=int(ckpt_args.get("ctx_dim", 64)),
        ctx_emb_dim=int(ckpt_args.get("ctx_emb_dim", emb_dim)),
        stage_channels=list(ckpt_args.get("ctx_stage_channels", [128, 256, 256])),
        stage_blocks=list(ckpt_args.get("ctx_stage_blocks", [2, 2, 2])),
        stage_res=list(ckpt_args.get("ctx_stage_res", [32, 16, 8])),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to latest.pt / epoch_xxxx.pt")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--demo_out", type=str, required=True)

    ap.add_argument("--use_norm", action="store_true", default=True, help="prefer data_norm if exists")
    ap.add_argument("--max_points", type=int, default=0,
                    help="生成/保存时每个样本最大点数（<=0: 用 ckpt te_max_sample_points；仍为0则全量）")
    ap.add_argument("--cd_points", type=int, default=0,
                    help="计算 CD 时使用点数（<=0: 用 max_points/全量）")

    ap.add_argument("--sample_steps", type=int, default=0, help="覆盖 ckpt 的 sample_steps（<=0 不覆盖）")
    ap.add_argument("--prior_std", type=float, default=0.0, help="覆盖 ckpt 的 prior_std（<=0 不覆盖）")
    ap.add_argument("--guidance_scale", type=float, default=None, help="覆盖 ckpt guidance_scale（不传则用 ckpt）")

    ap.add_argument("--no_ema", action="store_true", default=False, help="不用 EMA 权重")
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

    # sampler hyperparams: default from ckpt, allow override
    sample_steps = int(ckpt_args.get("sample_steps", 50))
    if args.sample_steps and args.sample_steps > 0:
        sample_steps = int(args.sample_steps)

    prior_std = float(ckpt_args.get("prior_std", 1.0))
    if args.prior_std and args.prior_std > 0:
        prior_std = float(args.prior_std)

    guidance_scale = float(ckpt_args.get("guidance_scale", 0.0))
    if args.guidance_scale is not None:
        guidance_scale = float(args.guidance_scale)

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
        cd_points = max_points  # 若仍 None，则后面用全量

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
    os.makedirs(out_gt, exist_ok=True)
    os.makedirs(out_pred, exist_ok=True)

    # scan test files
    files = find_h5_files(args.data_dir, args.split)
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 found. data_dir={args.data_dir} split={args.split}")

    print(f"[Demo] Found {len(files)} h5 shards. device={device} steps={sample_steps} prior_std={prior_std} "
          f"max_points={max_points} cd_points={cd_points} guidance_scale={guidance_scale}")

    cds: List[float] = []
    sample_idx = 0

    for fp in files:
        with h5py.File(fp, "r") as f:
            pts_key = _pick_points_key(f, prefer_norm=args.use_norm)
            cond_key = _pick_cond_key(f)

            pts_ds = f[pts_key]      # (B,N,3)
            cond_ds = f[cond_key]    # (B,D)

            B = int(pts_ds.shape[0])

            for i in tqdm(range(B), desc=f"[{os.path.basename(fp)}]", leave=False):
                sample_idx += 1
                name = f"{sample_idx:06d}.ply"

                # load gt & cond
                gt_np = np.asarray(pts_ds[i], dtype=np.float32)      # (N,3)
                cond_np = np.asarray(cond_ds[i], dtype=np.float32).reshape(-1)

                # optional subsample for generation/saving
                if max_points is not None:
                    gt_np = subsample_np(gt_np, max_points, np_rng)

                # save gt
                write_ply_xyz(os.path.join(out_gt, name), gt_np)

                # to torch
                gt = torch.from_numpy(gt_np)[None, ...].to(device=device, dtype=torch.float32)
                cond = torch.from_numpy(cond_np)[None, ...].to(device=device, dtype=torch.float32)

                # sample
                # (不使用 generator 参数，兼容老 torch)
                z = torch.randn_like(gt) * prior_std
                pred = euler_sampler(net, z, cond, steps=sample_steps, guidance_scale=guidance_scale)

                # save pred
                pred_np = pred[0].detach().cpu().numpy()
                write_ply_xyz(os.path.join(out_pred, name), pred_np)

                # CD
                pred_cd = pred
                gt_cd = gt
                if cd_points is not None and cd_points > 0:
                    pred_cd = subsample_torch_per_example(pred_cd, cd_points, np_rng)
                    gt_cd = subsample_torch_per_example(gt_cd, cd_points, np_rng)

                cd = float(chamfer_l2(pred_cd, gt_cd)[0].item())
                cds.append(cd)

    mean_cd = float(np.mean(cds)) if cds else float("nan")
    print(f"[Demo] Done. Samples={len(cds)} mean_CD={mean_cd:.8f}")

    # write summary
    os.makedirs(args.demo_out, exist_ok=True)
    with open(os.path.join(args.demo_out, "summary.json"), "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "data_dir": args.data_dir,
            "split": args.split,
            "backbone": backbone,
            "samples": len(cds),
            "mean_cd": mean_cd,
            "sample_steps": sample_steps,
            "prior_std": prior_std,
            "guidance_scale": guidance_scale,
            "max_points": max_points,
            "cd_points": cd_points,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
'''
export CUDA_VISIBLE_DEVICES=5
python demo_generate_tdcr.py \
  --ckpt runs/sim2_mlp_12_2_2W/ckpts/latest.pt \
  --data_dir datasets/sim_2m \
  --split test \
  --demo_out demo_out/sim2_mlp \
  --use_norm \
  --cd_points 4096

python demo_generate_tdcr.py \
  --ckpt runs/sim3_mlp_12_2_2W/ckpts/latest.pt \
  --data_dir datasets/sim_3m \
  --split test \
  --demo_out demo_out/sim3_mlp \
  --use_norm \
  --cd_points 4096

python demo_generate_tdcr.py \
  --ckpt runs/real2_mlp_12_2_2W/ckpts/latest.pt \
  --data_dir datasets/real_2m \
  --split test \
  --demo_out demo_out/real2_mlp \
  --use_norm \
  --cd_points 4096



python demo_generate_tdcr.py \
  --ckpt runs/sim2_hybrid_12_3/ckpts/latest.pt \
  --data_dir datasets/sim_2m \
  --split test \
  --demo_out demo_out/sim2_hybrid \
  --use_norm \
  --cd_points 4096

python demo_generate_tdcr.py \
  --ckpt runs/sim3_hybrid_12_3/ckpts/latest.pt \
  --data_dir datasets/sim_3m \
  --split test \
  --demo_out demo_out/sim3_hybrid \
  --use_norm \
  --cd_points 4096

python demo_generate_tdcr.py \
  --ckpt runs/real2_hybrid_12_3/ckpts/latest.pt \
  --data_dir datasets/real_2m \
  --split test \
  --demo_out demo_out/real2_hybrid \
  --use_norm \
  --cd_points 4096

python demo_generate_tdcr.py \
  --ckpt runs/sim5_with_base_mlp_12_18_2W/ckpts/latest.pt \
  --data_dir datasets/sim_5m_with_base \
  --split test \
  --demo_out demo_out/sim5_with_base_mlp \
  --use_norm \
  --cd_points 4096

'''