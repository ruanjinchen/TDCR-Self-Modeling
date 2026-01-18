#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_selfmodel_gs_dataset.py

将你在 MuJoCo 仿真中采集到的：
  - 多相机 RGB（2m_no_base/rgb/<cam_name>/<id>.<ext>）
  - segmentation mask（2m_no_base/rgb/<cam_name>/<id>_mask.png）
  - 电机控制（2m_no_base/motor/<id>.json）
  - 可选：点云（2m_no_base/pointcloud/<id>.ply）

整理成可直接喂给“3D Gaussian self-modeling”训练代码（RealData loader）的一套数据目录。

默认输出：
  out_root/
    images/<cam_name>/<id>.png        # RGBA（alpha=mask），RGB 默认会被 mask 抠成黑底
    points3D.txt
    joint.txt
    joint_zero.txt
    joint30.txt
    frame_ids.txt                     # 行号->原始采样 id 的映射（便于排查）
    info_all_train.json
    info_zero_train.json
    (可选) info_all_val.json / info_all_test.json

用法示例：
python make_selfmodel_gs_dataset.py \
  --xml tdcr2_no_base.xml \
  --motor_dir 2m_no_base/motor \
  --rgb_dir 2m_no_base/rgb \
  --pcd_dir 2m_no_base/pointcloud \
  --out_root datasets/tdcr2_selfmodel \
  --seed 42

训练时（注意 source_path 要带 .all/.zero 后缀）：
python train2.py -s datasets/tdcr2_selfmodel.zero -m out_tdcr --joints 3 --lambda_mask 0.1 -u 7000
python train2.py -s datasets/tdcr2_selfmodel.all  -m out_tdcr --joints 3 --lambda_mask 0.1 -k out_tdcr/chkpnt_7000.pth

依赖：
  pip install mujoco pillow numpy
  (可选) pip install open3d  # 如果你要从 ply 生成 points3D.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mujoco
except Exception:
    mujoco = None

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("需要 pillow: pip install pillow") from e

try:
    import open3d as o3d
except Exception:
    o3d = None


# -------------------------- helpers --------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_int_stem(p: Path) -> Optional[int]:
    """
    从文件名中抽取数字（支持 000123.json / 123.png / frame_123.json）。
    """
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else None


def load_ctrl_from_motor_json(p: Path) -> np.ndarray:
    obj = json.loads(p.read_text())
    if isinstance(obj, list):
        ctrl = obj
    elif isinstance(obj, dict):
        # 尽量兼容你 collect.py 的写法
        if "ctrl" in obj:
            ctrl = obj["ctrl"]
        elif "motor" in obj:
            ctrl = obj["motor"]
        elif "motor_ctrl" in obj:
            ctrl = obj["motor_ctrl"]
        else:
            raise KeyError(f"motor json 缺少 ctrl 字段: {p}")
    else:
        raise TypeError(f"motor json 格式异常: {p}")
    arr = np.asarray(ctrl, dtype=np.float32).reshape(-1)
    return arr


def normalize_ctrl(ctrl: np.ndarray, ctrlrange: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    ctrlrange: (D,2) [lo, hi]
    """
    lo = ctrlrange[:, 0].astype(np.float32)
    hi = ctrlrange[:, 1].astype(np.float32)
    denom = (hi - lo)
    denom[denom == 0] = 1.0
    u = (ctrl - lo) / denom
    if clip:
        u = np.clip(u, 0.0, 1.0)
    return u.astype(np.float32)


def fovy_to_fovx(fovy_rad: float, w: int, h: int) -> float:
    return float(2.0 * math.atan(math.tan(fovy_rad / 2.0) * (w / float(h))))


def quat_wxyz_to_R(q: Sequence[float]) -> np.ndarray:
    """
    MuJoCo MJCF quat 顺序是 [w x y z]。
    返回 3x3 rotation matrix（camera-to-world / local-to-world）。
    """
    w, x, y, z = map(float, q)
    # normalize
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n <= 0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w/n, x/n, y/n, z/n

    # standard quaternion to rotation
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


@dataclass
class CamInfo:
    name: str
    cam_id: int
    R_c2w: np.ndarray  # 3x3
    C_w: np.ndarray    # 3
    fovy_rad: float


def get_cameras_from_mujoco(xml_path: Path, cam_names: Optional[List[str]] = None) -> Tuple[List[CamInfo], np.ndarray]:
    """
    读取相机外参 + ctrlrange（用于电机归一化）。
    要求 mujoco python 可用。
    """
    if mujoco is None:
        raise RuntimeError("未安装 mujoco python：pip install mujoco")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # ctrlrange
    if not hasattr(model, "actuator_ctrlrange"):
        raise RuntimeError("MjModel 不含 actuator_ctrlrange（版本过旧？）")
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float32).copy()

    # cameras
    ncam = int(model.ncam)
    all_names: List[str] = []
    for cid in range(ncam):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        all_names.append(str(nm) if nm else f"cam{cid}")

    if cam_names:
        wanted = set(cam_names)
        cam_ids = [i for i, n in enumerate(all_names) if n in wanted]
        missing = sorted(list(wanted - set(all_names)))
        if missing:
            raise ValueError(f"指定的相机名在 xml 里找不到: {missing}. 可用相机: {all_names}")
    else:
        cam_ids = list(range(ncam))

    cams: List[CamInfo] = []
    for cid in cam_ids:
        nm = all_names[cid]
        C = np.asarray(data.cam_xpos[cid], dtype=np.float32).reshape(3)
        R = np.asarray(data.cam_xmat[cid], dtype=np.float32).reshape(3, 3)
        # ✅ MuJoCo/OpenGL -> COLMAP：翻转相机 y、z 轴
        R = R.copy()
        R[:, 1:3] *= -1
        fovy_deg = float(model.cam_fovy[cid]) if float(model.cam_fovy[cid]) > 0 else float(model.vis.global_.fovy)
        fovy_rad = float(np.deg2rad(fovy_deg))
        cams.append(CamInfo(name=nm, cam_id=cid, R_c2w=R, C_w=C, fovy_rad=fovy_rad))
    return cams, ctrlrange


def discover_camera_dirs(rgb_dir: Path) -> List[str]:
    cams = [p.name for p in rgb_dir.iterdir() if p.is_dir()]
    cams = sorted(cams)
    if not cams:
        raise RuntimeError(f"在 rgb_dir 下未找到相机子目录: {rgb_dir}")
    return cams


def find_rgb_path(cam_dir: Path, sid: int, preferred_ext: Optional[str]) -> Path:
    if preferred_ext:
        p = cam_dir / f"{sid:06d}.{preferred_ext}"
        if p.exists():
            return p
        # 也允许不补零
        p2 = cam_dir / f"{sid}.{preferred_ext}"
        if p2.exists():
            return p2
    # fallback: 任意匹配
    for ext in ("png", "jpg", "jpeg"):
        p = cam_dir / f"{sid:06d}.{ext}"
        if p.exists():
            return p
        p2 = cam_dir / f"{sid}.{ext}"
        if p2.exists():
            return p2
    raise FileNotFoundError(f"找不到 RGB: {cam_dir} sid={sid}")


def load_rgb_and_mask(rgb_path: Path, mask_path: Path, apply_mask_to_rgb: bool = True) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB")
    m = Image.open(mask_path).convert("L")
    if rgb.size != m.size:
        m = m.resize(rgb.size, resample=Image.NEAREST)

    rgb_np = np.array(rgb, dtype=np.uint8)
    m_np = np.array(m, dtype=np.uint8)
    alpha = (m_np > 127).astype(np.uint8) * 255

    if apply_mask_to_rgb:
        rgb_np = (rgb_np.astype(np.uint16) * (alpha[..., None].astype(np.uint16) // 255)).astype(np.uint8)

    rgba_np = np.concatenate([rgb_np, alpha[..., None]], axis=2)
    return Image.fromarray(rgba_np, mode="RGBA")


def write_joint_file(path: Path, joints: np.ndarray) -> None:
    """
    joints: (N, D) float32 in [0,1]
    格式必须是：
      [0.1 0.2 0.3]
    （不要逗号）
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in joints:
            f.write("[" + " ".join(f"{float(x):.6f}" for x in row.tolist()) + "]\n")


def write_points3d_from_ply(ply_path: Path, out_txt: Path, n_points: int = 50000, seed: int = 42) -> None:
    if o3d is None:
        raise RuntimeError("需要 open3d 才能从 ply 生成 points3D.txt：pip install open3d")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    col = np.asarray(pcd.colors, dtype=np.float32)
    if pts.size == 0:
        raise RuntimeError(f"点云为空: {ply_path}")

    if col.size == 0:
        col = np.zeros((pts.shape[0], 3), dtype=np.float32)

    # normalize color to 0..255 uint8
    if col.max() <= 1.0 + 1e-6:
        col_u8 = np.clip(col * 255.0, 0, 255).astype(np.uint8)
    else:
        col_u8 = np.clip(col, 0, 255).astype(np.uint8)

    # subsample
    rng = np.random.default_rng(seed)
    if pts.shape[0] > n_points:
        idx = rng.choice(pts.shape[0], size=n_points, replace=False)
        pts = pts[idx]
        col_u8 = col_u8[idx]

    with open(out_txt, "w", encoding="utf-8") as f:
        for i, (p, c) in enumerate(zip(pts, col_u8), start=1):
            f.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 0\n")


# -------------------------- main pipeline --------------------------
def build_infos(
    out_root: Path,
    cams: List[CamInfo],
    frame_ids: List[int],
    joint_map: Dict[int, int],
    image_paths: Dict[Tuple[int, str], Path],
    width: int,
    height: int,
) -> Dict[str, object]:
    images = []
    for sid in frame_ids:
        jid = int(joint_map[sid])
        for cam in cams:
            img_path = image_paths[(sid, cam.name)]
            R = cam.R_c2w
            C = cam.C_w
            t = (-R.T @ C).astype(np.float32)

            fovy = cam.fovy_rad
            fovx = fovy_to_fovx(fovy, width, height)

            images.append({
                "name": f"{cam.name}_{sid:06d}.png",
                "image_path": str(img_path.resolve()),
                "width": int(width),
                "height": int(height),
                "R": R.tolist(),
                "T": t.tolist(),
                "camera_id": int(cam.cam_id),
                "fovx": float(fovx),
                "fovy": float(fovy),
                "joint_id": int(jid),
            })
    return {"images": images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True, help="MuJoCo XML（用于读取相机位姿与 actuator ctrlrange）")
    ap.add_argument("--motor_dir", type=Path, required=True, help="motor json 目录（每个样本一个 json）")
    ap.add_argument("--rgb_dir", type=Path, required=True, help="rgb 目录（按相机名分子目录）")
    ap.add_argument("--pcd_dir", type=Path, default=None, help="点云 ply 目录（可选，用于生成 points3D.txt）")
    ap.add_argument("--out_root", type=Path, required=True, help="输出数据集根目录（不需要带 .all/.zero 后缀）")

    ap.add_argument("--rgb_ext", type=str, default=None, help="RGB 文件扩展名（png/jpg）。不填则自动探测")
    ap.add_argument("--mask_suffix", type=str, default="_mask.png", help="mask 文件名后缀（默认: _mask.png）")
    ap.add_argument("--apply_mask_to_rgb", action="store_true", help="把背景抠成黑（推荐）")
    ap.add_argument("--no_apply_mask_to_rgb", action="store_true", help="不改 RGB，只写 alpha（不推荐）")

    ap.add_argument("--cam_names", nargs="*", default=None, help="只处理指定相机名；不填则按 rgb_dir 下子目录全部处理")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.0)
    ap.add_argument("--test_frac", type=float, default=0.0)

    ap.add_argument("--min_mask_area_frac", type=float, default=0.0005,
                    help="过滤 mask 太小的帧（占比阈值）。默认 0.0005=0.05%%")
    ap.add_argument("--zero_sid", type=int, default=None, help="指定哪一帧当 zero。默认自动选 ctrl 最小的帧")
    ap.add_argument("--init_points", type=int, default=50000, help="points3D.txt 初始化点数")
    args = ap.parse_args()

    apply_mask = args.apply_mask_to_rgb and (not args.no_apply_mask_to_rgb)

    out_root = args.out_root
    ensure_dir(out_root)
    out_img_root = out_root / "images"
    ensure_dir(out_img_root)

    # cameras
    cam_dir_names = discover_camera_dirs(args.rgb_dir)
    if args.cam_names:
        use_cam_names = args.cam_names
    else:
        use_cam_names = cam_dir_names

    cams, ctrlrange = get_cameras_from_mujoco(args.xml, cam_names=use_cam_names)

    # frames
    motor_files = sorted([p for p in args.motor_dir.glob("*.json") if parse_int_stem(p) is not None],
                         key=lambda p: parse_int_stem(p))
    if not motor_files:
        raise RuntimeError(f"motor_dir 下没找到 *.json: {args.motor_dir}")

    # build frame list by checking RGB+mask existence
    frame_ids: List[int] = []
    ctrls_raw: List[np.ndarray] = []
    dropped_missing = 0
    dropped_smallmask = 0

    # probe resolution from first available rgb
    probe_sid = parse_int_stem(motor_files[0])
    assert probe_sid is not None
    probe_rgb = find_rgb_path(args.rgb_dir / cams[0].name, probe_sid, args.rgb_ext)
    w, h = Image.open(probe_rgb).size

    for mf in motor_files:
        sid = parse_int_stem(mf)
        if sid is None:
            continue

        ok = True
        # existence check for all cameras
        for cam in cams:
            cam_dir = args.rgb_dir / cam.name
            rgb_path = find_rgb_path(cam_dir, sid, args.rgb_ext)
            mask_path = cam_dir / f"{rgb_path.stem}{args.mask_suffix}" if args.mask_suffix.startswith(".") else cam_dir / f"{sid:06d}{args.mask_suffix}"
            # 更稳：优先按 sid+suffix
            mask_path2 = cam_dir / f"{sid:06d}{args.mask_suffix}"
            if mask_path2.exists():
                mask_path = mask_path2
            if not (rgb_path.exists() and mask_path.exists()):
                ok = False
                break
        if not ok:
            dropped_missing += 1
            continue

        # quick mask-area filter (use first camera)
        cam0_dir = args.rgb_dir / cams[0].name
        rgb0 = find_rgb_path(cam0_dir, sid, args.rgb_ext)
        m0 = cam0_dir / f"{sid:06d}{args.mask_suffix}"
        if not m0.exists():
            # fallback try stem-based
            m0 = cam0_dir / f"{rgb0.stem}{args.mask_suffix}"
        m_img = Image.open(m0).convert("L")
        if m_img.size != (w, h):
            m_img = m_img.resize((w, h), resample=Image.NEAREST)
        m_np = np.array(m_img, dtype=np.uint8)
        area = float((m_np > 127).sum()) / float(w * h)
        if area < args.min_mask_area_frac:
            dropped_smallmask += 1
            continue

        ctrl = load_ctrl_from_motor_json(mf)
        if ctrl.shape[0] != ctrlrange.shape[0]:
            raise ValueError(f"ctrl 维度={ctrl.shape[0]} 与 ctrlrange={ctrlrange.shape[0]} 不一致: {mf}")
        frame_ids.append(sid)
        ctrls_raw.append(ctrl)

    if not frame_ids:
        raise RuntimeError("没有任何可用帧（检查 rgb/mask 是否齐全，或调低 min_mask_area_frac）")

    # normalize ctrls to [0,1]
    ctrls = np.stack(ctrls_raw, axis=0)  # (N,D)
    ctrls_u = np.stack([normalize_ctrl(c, ctrlrange) for c in ctrls], axis=0)

    # map sid -> joint_id (0..N-1)
    frame_ids_sorted = sorted(frame_ids)
    joint_map = {sid: i for i, sid in enumerate(frame_ids_sorted)}

    # choose zero frame
    if args.zero_sid is not None:
        if args.zero_sid not in joint_map:
            raise ValueError(f"--zero_sid={args.zero_sid} 不在可用帧列表里")
        zero_sid = args.zero_sid
    else:
        # pick ctrl closest to zero (in normalized space)
        norms = np.linalg.norm(ctrls_u, axis=1)
        zero_sid = frame_ids[int(np.argmin(norms))]
    print(f"[export] frames kept: {len(frame_ids_sorted)} | dropped_missing={dropped_missing} dropped_smallmask={dropped_smallmask}")
    print(f"[export] zero_sid = {zero_sid:06d}")

    # write joint files
    joints_out = ctrls_u[np.array([frame_ids.index(sid) for sid in frame_ids_sorted], dtype=np.int64)]
    write_joint_file(out_root / "joint.txt", joints_out)

    # joint30: first 30 lines (or all if <30)
    n30 = min(30, joints_out.shape[0])
    write_joint_file(out_root / "joint30.txt", joints_out[:n30])

    # joint_zero: only one line, joint_id must be 0 in info_zero
    zero_jid = joint_map[zero_sid]
    write_joint_file(out_root / "joint_zero.txt", joints_out[zero_jid:zero_jid + 1])

    # write frame id mapping
    with open(out_root / "frame_ids.txt", "w", encoding="utf-8") as f:
        for sid in frame_ids_sorted:
            f.write(f"{sid:06d}\n")

    # export RGBA images
    image_paths: Dict[Tuple[int, str], Path] = {}
    for cam in cams:
        out_cam_dir = out_img_root / cam.name
        ensure_dir(out_cam_dir)

    for sid in frame_ids_sorted:
        for cam in cams:
            cam_dir = args.rgb_dir / cam.name
            rgb_in = find_rgb_path(cam_dir, sid, args.rgb_ext)
            mask_in = cam_dir / f"{sid:06d}{args.mask_suffix}"
            if not mask_in.exists():
                # fallback: try using rgb stem
                mask_in = cam_dir / f"{rgb_in.stem}{args.mask_suffix}"
            if not mask_in.exists():
                raise FileNotFoundError(f"找不到 mask: {mask_in}")

            out_img = out_img_root / cam.name / f"{sid:06d}.png"
            rgba = load_rgb_and_mask(rgb_in, mask_in, apply_mask_to_rgb=apply_mask)
            rgba.save(out_img)
            image_paths[(sid, cam.name)] = out_img

    # split train/val/test by frame
    rng = random.Random(args.seed)
    frames = frame_ids_sorted[:]
    rng.shuffle(frames)
    N = len(frames)
    n_test = int(round(N * float(args.test_frac)))
    n_val = int(round(N * float(args.val_frac)))
    n_test = min(n_test, N)
    n_val = min(n_val, N - n_test)
    test_frames = sorted(frames[:n_test])
    val_frames = sorted(frames[n_test:n_test + n_val])
    train_frames = sorted(frames[n_test + n_val:])

    # ------------------------------------------------------------
    # ✅ Avoid data leakage from stage1 (.zero):
    #    Make sure the selected zero_sid NEVER ends up in val/test.
    #    If it does, swap it back into train.
    #
    # Why:
    #   Stage1 always trains on `info_zero_train.json` (zero_sid).
    #   If zero_sid is in test/val, then stage1 has "seen" test/val.
    # ------------------------------------------------------------
    if zero_sid in test_frames:
        test_frames.remove(zero_sid)
        if len(train_frames) > 0:
            # swap one train frame to keep test size unchanged
            repl = train_frames.pop(0)
            test_frames.append(repl)
        print(f"[split-fix] zero_sid {zero_sid:06d} was in TEST -> moved to TRAIN")

    if zero_sid in val_frames:
        val_frames.remove(zero_sid)
        if len(train_frames) > 0:
            # swap one train frame to keep val size unchanged
            repl = train_frames.pop(0)
            val_frames.append(repl)
        print(f"[split-fix] zero_sid {zero_sid:06d} was in VAL -> moved to TRAIN")

    if zero_sid not in train_frames:
        train_frames.append(zero_sid)

    # keep deterministic order
    train_frames = sorted(train_frames)
    val_frames = sorted(val_frames)
    test_frames = sorted(test_frames)

    # build info jsons
    info_all_train = build_infos(out_root, cams, train_frames, joint_map, image_paths, w, h)
    (out_root / "info_all_train.json").write_text(json.dumps(info_all_train, indent=2), encoding="utf-8")

    if val_frames:
        info_all_val = build_infos(out_root, cams, val_frames, joint_map, image_paths, w, h)
        (out_root / "info_all_val.json").write_text(json.dumps(info_all_val, indent=2), encoding="utf-8")
    if test_frames:
        info_all_test = build_infos(out_root, cams, test_frames, joint_map, image_paths, w, h)
        (out_root / "info_all_test.json").write_text(json.dumps(info_all_test, indent=2), encoding="utf-8")

    # zero info: use only zero_sid images, joint_id must be 0 (because joint_zero.txt only has 1 line)
    # build by temporarily overriding joint_map
    zero_joint_map = {zero_sid: 0}
    info_zero = build_infos(out_root, cams, [zero_sid], zero_joint_map, image_paths, w, h)
    (out_root / "info_zero_train.json").write_text(json.dumps(info_zero, indent=2), encoding="utf-8")

    # points3D init
    if args.pcd_dir is not None:
        pcd_dir = args.pcd_dir
        # prefer zero frame point cloud if exists else first
        ply0 = pcd_dir / f"{zero_sid:06d}.ply"
        if not ply0.exists():
            ply0 = pcd_dir / f"{train_frames[0]:06d}.ply"
        if not ply0.exists():
            # fallback pick any ply
            ply_list = sorted(pcd_dir.glob("*.ply"))
            if not ply_list:
                raise RuntimeError(f"pcd_dir 下没有 ply: {pcd_dir}")
            ply0 = ply_list[0]
        print(f"[export] init points from: {ply0}")
        write_points3d_from_ply(ply0, out_root / "points3D.txt", n_points=int(args.init_points), seed=int(args.seed))
    else:
        print("[export] skip points3D.txt (no --pcd_dir). 你也可以稍后补上 points3D.txt。")

    print("\nDONE.")
    print(f"Dataset root: {out_root.resolve()}")
    print("Train with:")
    print(f"  python train2.py -s {out_root}.zero -m <outdir> --joints <D> --lambda_mask 0.1 -u 7000")
    print(f"  python train2.py -s {out_root}.all  -m <outdir> --joints <D> --lambda_mask 0.1 -k <outdir>/chkpnt_7000.pth")


if __name__ == "__main__":
    main()
'''
python make_selfmodel_gs_dataset.py \
  --xml tdcr2_no_base.xml \
  --motor_dir 2m_no_base/motor \
  --rgb_dir 2m_no_base/rgb \
  --pcd_dir 2m_no_base/pointcloud \
  --out_root 3dgs/2m_no_base \
  --seed 42 \
  --apply_mask_to_rgb --min_mask_area_frac 0.005 \
  --val_frac 0.1 --test_frac 0.1


python make_selfmodel_gs_dataset.py \
  --xml tdcr2_with_base.xml \
  --motor_dir 2m_with_base/motor \
  --rgb_dir 2m_with_base/rgb \
  --pcd_dir 2m_with_base/pointcloud \
  --out_root 3dgs/2m_with_base \
  --seed 42 \
  --apply_mask_to_rgb --min_mask_area_frac 0.005 \
  --val_frac 0.1 --test_frac 0.1

----------------------------------------------------------------------------------

python make_selfmodel_gs_dataset.py \
  --xml tdcr3_no_base.xml \
  --motor_dir 3m_no_base/motor \
  --rgb_dir 3m_no_base/rgb \
  --pcd_dir 3m_no_base/pointcloud \
  --out_root 3dgs/3m_no_base \
  --seed 42 \
  --apply_mask_to_rgb --min_mask_area_frac 0.005 \
  --val_frac 0.1 --test_frac 0.1


python make_selfmodel_gs_dataset.py \
  --xml tdcr3_with_base.xml \
  --motor_dir 3m_with_base/motor \
  --rgb_dir 3m_with_base/rgb \
  --pcd_dir 3m_with_base/pointcloud \
  --out_root 3dgs/3m_with_base \
  --seed 42 \
  --apply_mask_to_rgb --min_mask_area_frac 0.005 \
  --val_frac 0.1 --test_frac 0.1


----------------------------------------------------------------------------------

python make_selfmodel_gs_dataset.py \
  --xml tdcr5_no_base.xml \
  --motor_dir 5m_no_base/motor \
  --rgb_dir 5m_no_base/rgb \
  --pcd_dir 5m_no_base/pointcloud \
  --out_root 3dgs/5m_no_base \
  --seed 42 \
  --apply_mask_to_rgb --min_mask_area_frac 0.005 \
  --val_frac 0.1 --test_frac 0.1


python make_selfmodel_gs_dataset.py \
  --xml tdcr5_with_base.xml \
  --motor_dir 5m_with_base/motor \
  --rgb_dir 5m_with_base/rgb \
  --pcd_dir 5m_with_base/pointcloud \
  --out_root 3dgs/5m_with_base \
  --seed 42 \
  --apply_mask_to_rgb --min_mask_area_frac 0.005 \
  --val_frac 0.1 --test_frac 0.1


'''