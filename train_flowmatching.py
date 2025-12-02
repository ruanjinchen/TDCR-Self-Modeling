from __future__ import annotations
import os, argparse, json, random, re
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import VelocityNet, HybridMLP
from utils import (
    EMA, seed_all, init_distributed, cleanup_distributed, cosine_lr,
    save_point_cloud_ply, save_point_cloud_xyz, count_parameters
)
from datasets import get_datasets, init_np_seed

_USE_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

def make_autocast(enabled: bool, use_bf16: bool):
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    if _USE_NEW_AMP:
        return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
    else:
        from torch.cuda.amp import autocast as _autocast
        return _autocast(enabled=enabled, dtype=dtype)

def make_scaler(enabled: bool):
    if _USE_NEW_AMP:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    else:
        from torch.cuda.amp import GradScaler as _GradScaler
        return _GradScaler(enabled=enabled)

def sample_noise_like(x: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    return torch.randn_like(x) * std

def build_model(args) -> nn.Module:
    if getattr(args, "pf_backbone", "mlp") == "mlp":
        # 保持你现在的 MLP 完全不变
        m = VelocityNet(
            cond_dim=args.cond_dim,
            width=args.width,
            depth=args.depth,
            emb_dim=args.emb_dim,
            cfg_dropout_p=args.cfg_drop_p,
        )
    else:
        # HybridMLP：ContextNet(PVConv 金字塔) + VelocityNetWithContext
        if HybridMLP is None:
            raise ImportError("HybridMLP 未导入成功，请确认 models_with_hybrid.py "
                              "以及 third_party/pvcnn 安装无误。")

        m = HybridMLP(
            cond_dim=args.cond_dim,
            point_dim=3,  # TDCR 现在只用 xyz 点云，没有 RGB
            # ContextNet（复制你之前的默认）
            ctx_dim=args.ctx_dim,
            ctx_emb_dim=args.ctx_emb_dim,
            stage_channels=args.ctx_stage_channels,
            stage_blocks=args.ctx_stage_blocks,
            stage_res=args.ctx_stage_res,
            with_se=args.ctx_with_se,
            norm_type=args.ctx_norm,
            gn_groups=args.ctx_gn_groups,
            with_global=args.ctx_with_global,
            voxel_normalize=args.ctx_voxel_normalize,
            # t‑门控
            use_t_gate=True,
            t_gate_k=args.ctx_t_gate_k,
            t_gate_tau=args.ctx_t_gate_tau,
            pf_width=args.width,
            pf_depth=args.depth,
            pf_emb_dim=args.emb_dim,
            cfg_dropout_p=args.cfg_drop_p,
        )
    return m


@torch.no_grad()
def euler_sampler(model: nn.Module, x0: torch.Tensor, cond: Optional[torch.Tensor],
                  steps: int = 50, guidance_scale: float = 0.0, use_ema: bool = True) -> torch.Tensor:
    device = x0.device
    dt = 1.0 / steps
    x = x0
    net = model.module if hasattr(model, "module") else model
    backup = None
    if use_ema and hasattr(net, "ema_shadow"):
        backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(net.ema_shadow, strict=True)
    for i in range(steps):
        t = torch.full((x.shape[0],), (i + 0.5) * dt, device=device, dtype=x.dtype)
        v = net.guided_velocity(x, t, cond, guidance_scale=guidance_scale)  # ← 条件在采样中也用上了 :contentReference[oaicite:6]{index=6}
        x = x + v * dt
    if backup is not None:
        net.load_state_dict(backup, strict=True)
    return x

# --------- Chamfer Distance: prefer compiled chamfer_3D ---------
_CHAMFER_EXT = None
_CHAMFER_EXT_FAILED = False

def _load_chamfer_ext():
    """尝试只加载一次 chamfer_3D，失败则永远走 fallback。"""
    global _CHAMFER_EXT, _CHAMFER_EXT_FAILED
    if _CHAMFER_EXT is not None or _CHAMFER_EXT_FAILED:
        return _CHAMFER_EXT
    try:
        import importlib
        _CHAMFER_EXT = importlib.import_module("chamfer_3D")
        print("[Chamfer] Using compiled chamfer_3D extension.")
    except Exception as e:
        print(f"[Chamfer][WARN] chamfer_3D not available: {e}. Falling back to torch.cdist.")
        _CHAMFER_EXT_FAILED = True
        _CHAMFER_EXT = None
    return _CHAMFER_EXT

@torch.no_grad()
def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, N, 3)
    返回: (B,) 每个 batch 的 CD
    """
    ext = _load_chamfer_ext()
    if ext is not None:
        B, N, _ = pred.shape
        device = pred.device

        # 编译的 op 用 float32 + contiguous
        x = pred.contiguous().to(device=device, dtype=torch.float32)
        y = target.contiguous().to(device=device, dtype=torch.float32)

        d1 = torch.empty(B, N, device=device, dtype=torch.float32)
        d2 = torch.empty(B, N, device=device, dtype=torch.float32)
        i1 = torch.empty(B, N, device=device, dtype=torch.int32)
        i2 = torch.empty(B, N, device=device, dtype=torch.int32)

        _ = ext.forward(x, y, d1, d2, i1, i2)
        cd = d1.mean(dim=1) + d2.mean(dim=1)     # (B,)
        return cd.to(pred.dtype)

    # fallback: 原来的 torch.cdist 实现
    d2 = torch.cdist(pred, target, p=2).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


@torch.no_grad()
def save_vis_samples(args,
                     model,
                     epoch: int,
                     val_batch,
                     out_dir: str,
                     guidance_scale: float = 1.0,
                     use_ema: bool = True,
                     rank: int = 0,
                     writer: Optional[SummaryWriter] = None) -> float:
    """
    用当前 model + 一个 val batch 生成若干可视化样本，并计算 CD。
    返回 cd_mean（float）。
    """
    os.makedirs(out_dir, exist_ok=True)
    net = model.module if hasattr(model, "module") else model
    net.eval()
    with torch.inference_mode():
        pts = val_batch["test_points"].to(args.device, non_blocking=True)
        cond = val_batch.get("cond", None)
        if cond is not None:
            cond = cond.to(args.device, non_blocking=True).float()
        z = torch.randn_like(pts) * args.prior_std
        pred = euler_sampler(model, z, cond,
                             steps=args.sample_steps,
                             guidance_scale=guidance_scale,
                             use_ema=use_ema)
        cd = chamfer_l2(pred, pts)                 # (B,)
        cd_mean = float(cd.mean().detach().cpu())

        if rank == 0:
            print(f"[epoch {epoch:04d}] CD-L2(cond) mean = {cd_mean:.6f}")
            if writer is not None:
                writer.add_scalar("val/cd", cd_mean, epoch)

        for i in range(min(pts.shape[0], args.vis_count)):
            save_point_cloud_ply(pred[i], os.path.join(out_dir, f"ep{epoch:04d}_cond_{i}.ply"))
            save_point_cloud_ply(pts[i],  os.path.join(out_dir, f"ep{epoch:04d}_gt_{i}.ply"))

    return cd_mean


def train_one_epoch(model, net, opt, scaler, train_loader,
                    epoch, args, ema: EMA,
                    rank: int = 0, world_size: int = 1,
                    writer: Optional[SummaryWriter] = None):
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}") if rank == 0 else None
    loss_sum = 0.0
    n_step = 0
    for batch in train_loader:
        gnorm = None
        pts = batch["train_points"].to(args.device, non_blocking=True).float()
        cond = batch.get("cond", None)
        if cond is not None:
            cond = cond.to(args.device, non_blocking=True).float()
        B, N, _ = pts.shape
        z = sample_noise_like(pts, std=args.prior_std)
        t = torch.rand(B, device=args.device, dtype=pts.dtype)  # U[0,1]
        x_t = (1.0 - t)[:, None, None] * z + t[:, None, None] * pts
        target_v = (pts - z)
        cond_drop_mask = None
        if args.cfg_drop_p > 0.0 and args.cond_dim > 0 and cond is not None:
            drop = (torch.rand(B, device=args.device) < args.cfg_drop_p).to(pts.dtype)
            cond_drop_mask = drop[:, None]  # (B,1)
        with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
            pred_v = model(x_t, t, cond, cond_drop_mask=cond_drop_mask)  # ← 条件在训练前向中使用 :contentReference[oaicite:7]{index=7}
            loss = F.mse_loss(pred_v, target_v)
        scaler.scale(loss).backward()
        if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
            scaler.unscale_(opt)
            # 这里返回的是裁剪前的全局 grad norm
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(net)
        args.global_step += 1
        if args.use_cosine_lr:
            lr_now = cosine_lr(args.global_step, args.total_steps, args.lr,
                               min_lr=args.min_lr, warmup=args.warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        else:
            lr_now = opt.param_groups[0]["lr"]
        n_step += 1

        loss_v = float(loss.detach().cpu())
        loss_sum += loss_v

        # ---- TensorBoard: train metric ----
        if writer is not None and rank == 0:
            if args.tb_log_every <= 1 or (args.global_step % args.tb_log_every == 0):
                writer.add_scalar("train/loss", loss_v, args.global_step)
                writer.add_scalar("train/lr", float(lr_now), args.global_step)
                if gnorm is not None:
                    g_val = float(gnorm)
                    writer.add_scalar("train/grad_norm", g_val, args.global_step)
                    # 这个曲线 >0 表示这一 step 有触发 clip
                    if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                        writer.add_scalar(
                            "train/grad_clipped",
                            float(g_val > args.grad_clip_norm),
                            args.global_step
                        )

        if pbar is not None:
            pbar.set_postfix(loss=float(loss.detach().cpu()),
                             loss_avg=loss_sum / n_step,
                             lr=f"{lr_now:.2e}")
            pbar.update(1)
    if pbar is not None:
        pbar.close()
# =========================
# [Auto-Resume] 相关工具函数
# =========================

def _find_latest_ckpt(ckpt_dir: str):
    """返回 (path, epoch)；找不到则 (None, 0)。匹配 epoch_XXXX.pt。"""
    if not os.path.isdir(ckpt_dir):
        return None, 0
    best_ep, best_path = 0, None
    for fn in os.listdir(ckpt_dir):
        m = re.match(r"epoch_(\d+)\.pt$", fn)
        if m:
            ep = int(m.group(1))
            if ep > best_ep:
                best_ep = ep
                best_path = os.path.join(ckpt_dir, fn)
    return best_path, best_ep

def _find_resume_ckpt(ckpt_dir: str):
    """优先 latest.pt，其次最大 epoch_XXXX.pt。"""
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest, None  # epoch 从 ckpt 字典里读
    return _find_latest_ckpt(ckpt_dir)

def _move_opt_state_to_device(opt: torch.optim.Optimizer, device: torch.device):
    """把优化器状态里的 tensor 迁移到目标 device，防止恢复后因设备不一致报错。"""
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)

def _safe_load_ema(ema_obj: EMA, state_dict: dict, ref_model: nn.Module, device: torch.device):
    """
    用 ckpt 中的 EMA 覆盖当前 ema_obj.shadow 中存在的键，并迁移到 device。
    避免 KeyError，也兼容 CPU/GPU 混用的 ckpt。
    """
    cur = ema_obj.shadow
    ref_sd = ref_model.state_dict()
    for k in cur.keys():
        if k in state_dict:
            v = state_dict[k]
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                cur[k] = v.to(device=device, dtype=ref_sd[k].dtype)
    ema_obj.shadow = cur

def atomic_torch_save(obj, path: str):
    """原子写文件：先写到 .tmp，再 rename 覆盖。"""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _capture_rng_states():
    """保存 Python / NumPy / Torch / CUDA 的随机数状态，方便完整恢复。"""
    np_state = np.random.get_state()
    # 把 ndarray 转成 list，避免某些版本下的反序列化限制
    if isinstance(np_state, tuple) and len(np_state) >= 2 and hasattr(np_state[1], "tolist"):
        np_state = (np_state[0], np_state[1].tolist(), *np_state[2:])
    return {
        "torch": torch.get_rng_state(),
        "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np_state,
        "python": random.getstate(),
    }

def _restore_rng_states(rng):
    """恢复随机数状态；失败时给个 warning 但不打断训练。"""
    if rng is None:
        return
    try:
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(rng["torch"])
        if "cuda_all" in rng and rng["cuda_all"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda_all"])
        if "numpy" in rng and rng["numpy"] is not None:
            np_state = rng["numpy"]
            if isinstance(np_state, tuple) and len(np_state) >= 2 and isinstance(np_state[1], list):
                arr = np.array(np_state[1], dtype=np.uint32)
                np_state = (np_state[0], arr, *np_state[2:])
            np.random.set_state(np_state)
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])
    except Exception as e:
        print(f"[WARN] Failed to restore RNG states: {e}")

def main():
    parser = argparse.ArgumentParser("MeanFlow training (Gaussian -> point cloud)")
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root (contains train/val/test shards).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)

    # Dataset-specific knobs (forwarded to get_datasets)
    parser.add_argument("--tdcr_use_norm", action="store_true", default=True, help="true 时优先使用 data_norm")
    parser.add_argument("--tr_max_sample_points", type=int, default=2048)
    parser.add_argument("--te_max_sample_points", type=int, default=2048)
    parser.add_argument("--cond_mode", type=str, default="motors")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--train_subset_seed", type=int, default=0)

    # Model
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--cfg_drop_p", type=float, default=0.0)
    # 选择点流骨干：mlp（原始 VelocityNet） 或 hybrid（PVConv Context + 逐点 MLP）
    parser.add_argument("--pf_backbone", type=str, default="mlp",
                        choices=["mlp", "hybrid"],
                        help="point-flow backbone: 'mlp' or 'hybrid'")

    # Hybrid 上下文分支（ContextNet）超参
    parser.add_argument("--ctx_dim", type=int, default=64)
    parser.add_argument("--ctx_emb_dim", type=int, default=256)
    parser.add_argument("--ctx_stage_channels", type=int, nargs="+",
                        default=[128, 256, 256])
    parser.add_argument("--ctx_stage_blocks", type=int, nargs="+",
                        default=[2, 2, 2])
    parser.add_argument("--ctx_stage_res", type=int, nargs="+",
                        default=[32, 16, 8])
    parser.add_argument("--ctx_with_se", action="store_true", default=True)
    parser.add_argument("--ctx_norm", type=str, default="group",
                        choices=["group", "batch", "syncbn", "none"])
    parser.add_argument("--ctx_gn_groups", type=int, default=32)
    parser.add_argument("--ctx_with_global", action="store_true", default=True)
    parser.add_argument("--ctx_voxel_normalize", action="store_true", default=True)

    # t-gate（决定什么时候更多依赖 PVConv 上下文）
    parser.add_argument("--ctx_t_gate_tau", type=float, default=0.95)
    parser.add_argument("--ctx_t_gate_k", type=float, default=5.0)

    # Optim
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--use_cosine_lr", action="store_true", default=True)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Flow / sampling / I/O
    parser.add_argument("--prior_std", type=float, default=1.0)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10,
                        help="多少个 epoch 存一个编号 checkpoint（epoch_xxxx.pt）")
    parser.add_argument("--val_every", type=int, default=0,
                        help="多少个 epoch 做一次 val+可视化+写 val/CD；<=0 表示跟 save_every 一样")
    parser.add_argument("--vis_count", type=int, default=16)
    parser.add_argument("--save_uncond", action="store_true", default=True)
    parser.add_argument("--guidance_scale", type=float, default=0.0)

    # System / I/O
    parser.add_argument("--out_dir", type=str, default="./runs/fm_tdcr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=False)

    # TensorBoard
    parser.add_argument("--no_tb", action="store_true",
                        help="关闭 TensorBoard 日志（默认开启）")
    parser.add_argument("--tb_log_dir", type=str, default=None,
                        help="TensorBoard 日志目录（默认 out_dir/tb）")
    parser.add_argument("--tb_log_every", type=int, default=50,
                        help="多少个 global_step 记录一次 train 指标；<=1 表示每步记录")

    args = parser.parse_args()

    # ddp init
    is_dist, rank, world_size, local_rank = init_distributed()
    args.is_distributed = is_dist
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    # val_every 默认跟 save_every 一致
    if args.val_every is None or args.val_every <= 0:
        args.val_every = args.save_every
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    seed_all(args.seed + rank)

    # datasets（内部会设置 args.cond_dim）
    tr_ds, te_ds = get_datasets(args)   # ← 数据集把 cond 组装好，并给出 cond_dim :contentReference[oaicite:8]{index=8}

    # samplers & loaders
    if is_dist:
        tr_sampler = DistributedSampler(tr_ds, shuffle=True, drop_last=True)
        te_sampler = DistributedSampler(te_ds, shuffle=False, drop_last=False)
    else:
        tr_sampler = None
        te_sampler = None

    train_loader = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=(tr_sampler is None),
        sampler=tr_sampler, num_workers=args.num_workers, drop_last=True,
        pin_memory=True, worker_init_fn=init_np_seed
    )
    val_loader = DataLoader(
        te_ds, batch_size=args.batch_size, shuffle=False,
        sampler=te_sampler, num_workers=max(1, args.num_workers // 2), drop_last=False,
        pin_memory=True, worker_init_fn=init_np_seed
    )

    net = build_model(args).to(args.device)
    if args.compile:
        net = torch.compile(net)

    ema = EMA(net, decay=0.999)
    net.ema_shadow = ema.shadow

    model = net
    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(net, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=False)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(enabled=args.amp)

    if rank == 0:
        print(f"Model params: {count_parameters(net)/1e6:.2f} M")
        print(f"cond_dim={args.cond_dim}  width={args.width} depth={args.depth} emb_dim={args.emb_dim}")
        print(f"WorldSize={world_size}  Device={args.device}")

    args.total_steps = args.epochs * max(1, len(train_loader))
    args.global_step = 0

    ckpt_dir = os.path.join(args.out_dir, "ckpts")
    start_epoch = 1

    ckpt_path, ckpt_ep = _find_resume_ckpt(ckpt_dir)
    if ckpt_path is not None:
        if rank == 0:
            print(f"[Resume] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)

        # 恢复模型
        if "model" in ckpt:
            net.load_state_dict(ckpt["model"], strict=True)

        # 恢复优化器
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
            _move_opt_state_to_device(opt, args.device)

        # 恢复 EMA
        if ema is not None and "ema" in ckpt:
            _safe_load_ema(ema, ckpt["ema"], net, args.device)

        # 恢复 scaler（若有）
        if "scaler" in ckpt and scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                if rank == 0:
                    print(f"[WARN] Failed to load scaler from ckpt: {e}")

        # epoch / global_step / RNG
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "global_step" in ckpt:
            args.global_step = int(ckpt["global_step"])
        _restore_rng_states(ckpt.get("rng_state", None))

        if rank == 0:
            print(f"[Resume] Start from epoch {start_epoch}, global_step={args.global_step}")
    else:
        if rank == 0:
            print("[Resume] No checkpoint found, training from scratch.")

    val_iter = iter(val_loader)
    try:
        val_batch = next(val_iter)
    except StopIteration:
        val_batch = next(iter(val_loader))

    writer = None
    if (rank == 0) and (not args.no_tb):
        tb_dir = args.tb_log_dir if args.tb_log_dir is not None else os.path.join(args.out_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    if rank == 0:
        print(f"Model params: {count_parameters(net)/1e6:.2f} M")
        print(f"cond_dim={args.cond_dim}  width={args.width} depth={args.depth} emb_dim={args.emb_dim}")
        print(f"WorldSize={world_size}  Device={args.device}")


    try:
        for ep in range(start_epoch, args.epochs + 1):
            if is_dist and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(ep)
            train_one_epoch(model, net, opt, scaler, train_loader, ep, args, ema, rank, world_size, writer=writer)
            if is_dist and dist.is_initialized():
                dist.barrier()
            # ---- 2) 保存 checkpoint：每个 epoch 都有 latest.pt，另外按 save_every 存历史 ----
            if rank == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_obj = {
                    "epoch": ep,
                    "model": net.state_dict(),
                    "ema": ema.shadow if ema is not None else None,
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "global_step": args.global_step,
                    "rng_state": _capture_rng_states(),
                    "args": vars(args),
                }
                # latest
                atomic_torch_save(ckpt_obj, os.path.join(ckpt_dir, "latest.pt"))
                # 历史归档
                if (ep % args.save_every == 0) or (ep == args.epochs):
                    atomic_torch_save(ckpt_obj, os.path.join(ckpt_dir, f"epoch_{ep:04d}.pt"))

            if is_dist and dist.is_initialized():
                dist.barrier()

            # ---- 3) val + 可视化 + 写 val/CD ----
            do_val = (ep % args.val_every == 0) or (ep == args.epochs)
            if do_val and rank == 0:
                vis_dir = os.path.join(args.out_dir, "samples", f"epoch_{ep:04d}")
                cd_mean = save_vis_samples(
                    args, model, ep, val_batch, vis_dir,
                    guidance_scale=args.guidance_scale,
                    use_ema=True,
                    rank=rank,
                    writer=writer,
                )

            if is_dist and dist.is_initialized():
                dist.barrier()
    finally:
        if writer is not None and rank == 0:
            writer.close()
        cleanup_distributed()

if __name__ == "__main__":
    main()




'''

export CUDA_VISIBLE_DEVICES=4,5
torchrun --standalone --nproc_per_node=2 --master_port=29511 \

python train_flowmatching.py \
  --data_dir datasets/sim_2m \
  --batch_size 8 --epochs 500 --save_every 20 \
  --emb_dim 64 --width 256 --depth 4 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --out_dir runs/sim2_mlp_12_2_2W

python train_flowmatching.py \
  --data_dir datasets/sim_3m \
  --batch_size 8 --epochs 500 --save_every 20 \
  --emb_dim 64 --width 256 --depth 4 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --out_dir runs/sim3_mlp_12_2_2W

python train_flowmatching.py \
  --data_dir datasets/real_2m \
  --batch_size 8 --epochs 500 --save_every 20 \
  --emb_dim 64 --width 256 --depth 4 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --out_dir runs/real2_mlp_12_2_2W


  
export CUDA_VISIBLE_DEVICES=4,5
torchrun --standalone --nproc_per_node=2 --master_port=29511 \
python train_flowmatching.py \
  --data_dir datasets/sim_2m \
  --batch_size 16 --epochs 500 --save_every 10 --val_every 10 \
  --tr_max_sample_points 20000 --te_max_sample_points 4096 \
  --cond_mode motors --lr 6e-4 \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 64 96 96 \
  --ctx_stage_blocks 1 1 1 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --out_dir runs/sim2_hybrid__12_1

python train_flowmatching.py \
  --data_dir datasets/sim_3m \
  --batch_size 16 --epochs 500 --save_every 10 --val_every 10 \
  --tr_max_sample_points 4096 --te_max_sample_points 4096 \
  --cond_mode motors --lr 6e-4 \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 64 96 96 \
  --ctx_stage_blocks 1 1 1 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --out_dir runs/sim3_hybrid__12_1

'''