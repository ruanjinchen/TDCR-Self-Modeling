#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_selfmodel_dnerf_dataset.py

目标：
  在“尽量不改训练代码(main_dnerf.py / rob_dnerf/provider.py)”的前提下，
  把你采集的仿真数据（多相机 RGB + mask + motor json + MuJoCo XML）整理成
  rob_dnerf/provider.py 的 real 模式能直接读取的格式。

rob_dnerf/provider.py(real 模式) 的关键要求：
  - 训练时传入的 path 形如：<dataset_root>.<joint_type>
      joint_type 必须是：all / 0 / 30d
    例如：/abs/path/datasets/tdcr2.all
  - dataset_root 目录下必须有：
      info_<joint_type>_train.json
      joint.txt
      joint_zero.txt
      joint30.txt
  - info json 顶层必须有 "images"（不是 "frames"）
    每个条目至少要有：
      image_path: RGBA png 的路径（alpha 是 mask）
      R: 3x3 (c2w rotation)  （provider 内部会用 R.T 当 w2c rotation）
      T: 3   (w2c translation)
      joint_id: 索引 joint*.txt 的行号
    并建议提供：
      fov_x / fov_y (弧度)，否则 provider 会用默认值 0.92 / 0.71

输入数据假设（与你现有 make_selfmodel_gs_dataset.py 一致）：
  - motor_dir/<sid>.json
  - rgb_dir/<cam_name>/<sid>.<png/jpg...>
  - rgb_dir/<cam_name>/<sid>_mask.png (默认后缀 _mask.png，可改)

用法示例：
python make_selfmodel_dnerf_dataset.py \
  --xml tdcr2_no_base.xml \
  --motor_dir 2m_no_base/motor \
  --rgb_dir 2m_no_base/rgb \
  --out_root dnerf/2m_no_base \
  --seed 42 \
  --apply_mask_to_rgb \
  --min_mask_area_frac 0.005 \
  --test_frac 0.1 \
  --joint_out norm_pm1

训练（不要用 -O；也不要开 --cuda_ray）：
python main_dnerf.py dnerf/2m_no_base.0   --workspace ws_tdcr2 --joints_num 3 --fp16 --preload --min_near 0.1 --break_iter 0
python main_dnerf.py dnerf/2m_no_base.all --workspace ws_tdcr2 --joints_num 3 --fp16 --preload --min_near 0.1 --break_iter 0

依赖：
  pip install mujoco pillow numpy
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mujoco  # type: ignore
except Exception:
    mujoco = None

try:
    from PIL import Image  # type: ignore
except Exception as e:
    raise RuntimeError("需要 pillow: pip install pillow") from e


# -------------------------- small utils --------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_int_stem(p: Path) -> Optional[int]:
    """从文件名中抽取数字（支持 000123.json / 123.png / frame_123.json）。"""
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else None


def load_ctrl_from_motor_json(p: Path) -> np.ndarray:
    """兼容 list / dict 多种 motor json。"""
    obj = json.loads(p.read_text())
    if isinstance(obj, list):
        ctrl = obj
    elif isinstance(obj, dict):
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


def normalize_ctrl_to_01(ctrl: np.ndarray, ctrlrange: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    ctrlrange: (D,2) [lo, hi]
    返回 u in [0,1]
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


def find_rgb_path(cam_dir: Path, sid: int, preferred_ext: Optional[str]) -> Path:
    """优先按 preferred_ext 搜索；否则遍历常见扩展名。支持 000001 和 1 两种命名。"""
    if preferred_ext:
        for fmt in (f"{sid:06d}.{preferred_ext}", f"{sid}.{preferred_ext}"):
            p = cam_dir / fmt
            if p.exists():
                return p
    for ext in ("png", "jpg", "jpeg"):
        for fmt in (f"{sid:06d}.{ext}", f"{sid}.{ext}"):
            p = cam_dir / fmt
            if p.exists():
                return p
    raise FileNotFoundError(f"找不到 RGB: {cam_dir} sid={sid}")


def find_mask_path(cam_dir: Path, sid: int, mask_suffix: str, rgb_stem: str) -> Path:
    """
    默认 mask_suffix 形如 "_mask.png"，优先尝试：
      000001_mask.png -> 1_mask.png -> <rgb_stem>_mask.png
    """
    cand = [
        cam_dir / f"{sid:06d}{mask_suffix}",
        cam_dir / f"{sid}{mask_suffix}",
        cam_dir / f"{rgb_stem}{mask_suffix}",
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 mask: sid={sid} cam_dir={cam_dir} mask_suffix={mask_suffix}")


def load_rgba(rgb_path: Path, mask_path: Path, apply_mask_to_rgb: bool = False) -> Image.Image:
    """
    读入 RGB 与 mask(L) -> 输出 RGBA(PNG)。
    alpha: 0/255（二值）
    """
    rgb = Image.open(rgb_path).convert("RGB")
    m = Image.open(mask_path).convert("L")
    if m.size != rgb.size:
        m = m.resize(rgb.size, resample=Image.NEAREST)

    rgb_np = np.array(rgb, dtype=np.uint8)
    m_np = np.array(m, dtype=np.uint8)
    alpha = (m_np > 127).astype(np.uint8) * 255

    if apply_mask_to_rgb:
        rgb_np = (rgb_np.astype(np.uint16) * (alpha[..., None].astype(np.uint16) // 255)).astype(np.uint8)

    rgba = np.concatenate([rgb_np, alpha[..., None]], axis=2)
    return Image.fromarray(rgba, mode="RGBA")


def write_joint_file(path: Path, joints: np.ndarray) -> None:
    """
    joints: (N, D)
    格式必须是：
      [0.1 0.2 0.3]
    （不要逗号）
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in joints:
            f.write("[" + " ".join(f"{float(x):.6f}" for x in row.tolist()) + "]\n")


# -------------------------- MuJoCo camera extraction --------------------------
@dataclass
class CamInfo:
    name: str
    cam_id: int
    R_c2w: np.ndarray  # 3x3
    C_w: np.ndarray    # 3
    fovy_rad: float


def get_cameras_from_mujoco(
    xml_path: Path,
    wanted_cam_names: List[str],
    flip_yz: bool = True,
) -> Tuple[List[CamInfo], np.ndarray]:
    """
    读取相机外参 + ctrlrange（用于 motor->joint 归一化/映射）
    flip_yz=True：
      把 MuJoCo(OpenGL-ish) 的相机坐标系转换到 OpenCV/COLMAP 风格（翻转相机 y,z 轴）。
      provider.py 在读 pose 后还会做一次 pose[:3,1:3]*=-1。
      如果你发现相机方向不对（比如左右/上下翻），可以试试 --no_flip_yz。
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

    # camera name lookup
    ncam = int(model.ncam)
    all_names: List[str] = []
    for cid in range(ncam):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        all_names.append(str(nm) if nm else f"cam{cid}")

    missing = sorted(list(set(wanted_cam_names) - set(all_names)))
    if missing:
        raise ValueError(
            f"指定的相机名在 xml 里找不到: {missing}\n"
            f"xml 中可用相机: {all_names}\n"
            f"rgb_dir 下的相机子目录: {wanted_cam_names}\n"
            f"如命名不一致，请用 --cam_names 手工指定与 xml 一致的名字。"
        )

    cams: List[CamInfo] = []
    for cid, nm in enumerate(all_names):
        if nm not in wanted_cam_names:
            continue

        C = np.asarray(data.cam_xpos[cid], dtype=np.float32).reshape(3)
        R = np.asarray(data.cam_xmat[cid], dtype=np.float32).reshape(3, 3)

        if flip_yz:
            R = R.copy()
            R[:, 1:3] *= -1

        # MuJoCo 的 cam_fovy 是度；若为 0，用 global fovy
        fovy_deg = float(model.cam_fovy[cid]) if float(model.cam_fovy[cid]) > 0 else float(model.vis.global_.fovy)
        fovy_rad = float(np.deg2rad(fovy_deg))

        cams.append(CamInfo(name=nm, cam_id=cid, R_c2w=R, C_w=C, fovy_rad=fovy_rad))

    cams = sorted(cams, key=lambda c: c.name)
    return cams, ctrlrange


def discover_camera_dirs(rgb_dir: Path) -> List[str]:
    cams = [p.name for p in rgb_dir.iterdir() if p.is_dir()]
    cams = sorted(cams)
    if not cams:
        raise RuntimeError(f"在 rgb_dir 下未找到相机子目录: {rgb_dir}")
    return cams


# -------------------------- info json builders --------------------------
def build_info_json(
    cams: List[CamInfo],
    sids: List[int],
    sid_to_joint_id: Dict[int, int],
    sid_cam_to_out_img: Dict[Tuple[int, str], Path],
    w: int,
    h: int,
    fov_x: float,
    fov_y: float,
    relative_paths: bool,
) -> Dict[str, object]:
    images: List[Dict[str, object]] = []
    for sid in sids:
        jid = int(sid_to_joint_id[sid])
        for cam in cams:
            out_img = sid_cam_to_out_img[(sid, cam.name)]
            img_path = out_img
            if relative_paths:
                # 相对路径相对于数据集根目录(out_root)
                # 注意：训练时你需要在能解析该相对路径的工作目录下运行（一般不如绝对路径稳）。
                img_path_str = img_path.as_posix()
            else:
                img_path_str = str(img_path.resolve())

            # 这里的 R/T 设计与 provider.py 完全对齐：
            # provider 用 w2c[:3,:3] = R.T, w2c[:3,3]=T -> pose=inv(w2c) -> pose[:3,1:3]*=-1
            R = cam.R_c2w.astype(np.float32)
            C = cam.C_w.astype(np.float32)
            T = (-R.T @ C).astype(np.float32)

            images.append({
                "image_path": img_path_str,
                "R": R.tolist(),
                "T": T.tolist(),
                "joint_id": jid,
                "fov_x": float(fov_x),
                "fov_y": float(fov_y),
                "camera_id": int(cam.cam_id),
                "width": int(w),
                "height": int(h),
            })

    # provider 会优先读顶层的 h/w（如果存在）
    return {
        "h": int(h),
        "w": int(w),
        "images": images,
    }


# -------------------------- main --------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--xml", type=Path, required=True, help="MuJoCo XML（用于读取相机位姿与 actuator ctrlrange）")
    ap.add_argument("--motor_dir", type=Path, required=True, help="motor json 目录（每个样本一个 json）")
    ap.add_argument("--rgb_dir", type=Path, required=True, help="rgb 目录（按相机名分子目录）")
    ap.add_argument("--out_root", type=Path, required=True, help="输出数据集根目录（不要带 .all/.0/.30d 后缀）")

    ap.add_argument("--rgb_ext", type=str, default=None, help="RGB 文件扩展名（png/jpg）。不填则自动探测")
    ap.add_argument("--mask_suffix", type=str, default="_mask.png", help="mask 文件名后缀（默认: _mask.png）")

    ap.add_argument("--cam_names", nargs="*", default=None,
                    help="只处理指定相机名（必须与 xml 中的 camera name 一致）；不填则按 rgb_dir 下子目录全部处理")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.0, help="按 frame(sid) 划分 test 的比例（训练代码默认不读 test）")

    ap.add_argument("--min_mask_area_frac", type=float, default=0.0005,
                    help="过滤 mask 太小的帧（占比阈值）。默认 0.0005=0.05%%")

    ap.add_argument("--zero_sid", type=int, default=None,
                    help="指定哪一帧当作 joint_zero。默认：选择 motor 归一化后 L2 最小的帧（更接近 ctrlrange 的下界）")

    ap.add_argument("--apply_mask_to_rgb", action="store_true",
                    help="把背景抠成黑（RGB 乘 alpha）。alpha 仍会写入。")
    ap.add_argument("--relative_paths", action="store_true",
                    help="info json 里写相对 image_path（默认写绝对路径，更不容易踩路径坑）")

    ap.add_argument("--no_flip_yz", action="store_true",
                    help="不要对 MuJoCo 相机做 y/z 轴翻转（如果你发现相机方向不对可尝试）。")

    ap.add_argument(
        "--joint_out",
        type=str,
        default="norm_pm1",
        choices=["raw", "norm01", "norm_pm1"],
        help=(
            "写入 joint*.txt 的关节值模式：\n"
            "  raw      : 直接写 motor json 里的 ctrl（provider 还会除以 pi）\n"
            "  norm01   : 把 ctrl 映射到 [0,1]，再乘 pi 写入（provider /pi 后得到 [0,1]）\n"
            "  norm_pm1 : 把 ctrl 映射到 [0,1]，再映射到 [-1,1]，再乘 pi 写入（provider /pi 后得到 [-1,1]）【推荐】"
        ),
    )

    ap.add_argument("--joint30_size", type=int, default=30, help="joint30.txt 取多少个关节配置（默认 30）")

    args = ap.parse_args()

    out_root = args.out_root
    if "." in out_root.name:
        print(f"[WARN] out_root 的最后一级目录名包含 '.'：{out_root.name}。"
              f"rob_dnerf/provider.py 会用 path.split('.') 解析 joint_type，建议避免路径里出现 '.'。")

    ensure_dir(out_root)
    out_img_root = out_root / "images"
    ensure_dir(out_img_root)

    # camera names
    rgb_cam_dirs = discover_camera_dirs(args.rgb_dir)
    wanted_cam_names = args.cam_names if args.cam_names is not None and len(args.cam_names) > 0 else rgb_cam_dirs

    # cameras + ctrlrange
    cams, ctrlrange = get_cameras_from_mujoco(args.xml, wanted_cam_names=wanted_cam_names, flip_yz=(not args.no_flip_yz))
    if not cams:
        raise RuntimeError("没有找到任何相机（检查 --cam_names / rgb_dir / xml）")

    # gather motor files
    motor_files = sorted([p for p in args.motor_dir.glob("*.json") if parse_int_stem(p) is not None],
                         key=lambda p: parse_int_stem(p))
    if not motor_files:
        raise RuntimeError(f"motor_dir 下没找到 *.json: {args.motor_dir}")

    # probe resolution from first available rgb
    first_sid = parse_int_stem(motor_files[0])
    assert first_sid is not None
    probe_rgb = find_rgb_path(args.rgb_dir / cams[0].name, first_sid, args.rgb_ext)
    w, h = Image.open(probe_rgb).size

    # filter frames with complete rgb+mask for all cams, and enough mask area
    kept_sids: List[int] = []
    sid_to_ctrl_raw: Dict[int, np.ndarray] = {}
    sid_to_u01: Dict[int, np.ndarray] = {}

    dropped_missing = 0
    dropped_smallmask = 0

    for mf in motor_files:
        sid = parse_int_stem(mf)
        if sid is None:
            continue

        # check all cameras exist
        ok = True
        for cam in cams:
            cam_dir = args.rgb_dir / cam.name
            try:
                rgb_in = find_rgb_path(cam_dir, sid, args.rgb_ext)
                _ = find_mask_path(cam_dir, sid, args.mask_suffix, rgb_in.stem)
            except FileNotFoundError:
                ok = False
                break
        if not ok:
            dropped_missing += 1
            continue

        # quick mask area check on first cam
        cam0_dir = args.rgb_dir / cams[0].name
        rgb0 = find_rgb_path(cam0_dir, sid, args.rgb_ext)
        m0 = find_mask_path(cam0_dir, sid, args.mask_suffix, rgb0.stem)
        m_img = Image.open(m0).convert("L")
        if m_img.size != (w, h):
            m_img = m_img.resize((w, h), resample=Image.NEAREST)
        m_np = np.array(m_img, dtype=np.uint8)
        area = float((m_np > 127).sum()) / float(w * h)
        if area < float(args.min_mask_area_frac):
            dropped_smallmask += 1
            continue

        ctrl_raw = load_ctrl_from_motor_json(mf)
        if ctrl_raw.shape[0] != ctrlrange.shape[0]:
            raise ValueError(
                f"ctrl 维度={ctrl_raw.shape[0]} 与 ctrlrange={ctrlrange.shape[0]} 不一致: {mf}\n"
                f"（通常表示 motor json 记录的 ctrl 与 xml 中 actuator 数不一致）"
            )

        u01 = normalize_ctrl_to_01(ctrl_raw, ctrlrange, clip=True)

        kept_sids.append(sid)
        sid_to_ctrl_raw[sid] = ctrl_raw
        sid_to_u01[sid] = u01

    if not kept_sids:
        raise RuntimeError("没有任何可用帧（检查 rgb/mask 是否齐全，或调低 min_mask_area_frac）")

    kept_sids = sorted(kept_sids)

    # choose zero sid
    if args.zero_sid is not None:
        if args.zero_sid not in sid_to_u01:
            raise ValueError(f"--zero_sid={args.zero_sid} 不在可用帧列表里")
        zero_sid = int(args.zero_sid)
    else:
        norms = [(sid, float(np.linalg.norm(sid_to_u01[sid]))) for sid in kept_sids]
        norms.sort(key=lambda x: x[1])
        zero_sid = int(norms[0][0])

    # build joint matrices in the SAME order as kept_sids
    D = int(ctrlrange.shape[0])
    joints_all = np.zeros((len(kept_sids), D), dtype=np.float32)

    for i, sid in enumerate(kept_sids):
        ctrl_raw = sid_to_ctrl_raw[sid]
        u01 = sid_to_u01[sid]
        if args.joint_out == "raw":
            j = ctrl_raw
        elif args.joint_out == "norm01":
            # 写入 u*pi，使得 provider 读入后 /pi -> u in [0,1]
            j = u01 * np.pi
        elif args.joint_out == "norm_pm1":
            # 写入 ((u*2-1)*pi)，使得 provider 读入后 /pi -> (u*2-1) in [-1,1]
            j = (u01 * 2.0 - 1.0) * np.pi
        else:
            raise ValueError(f"unknown joint_out: {args.joint_out}")
        joints_all[i] = j.astype(np.float32)

    # joint30: 前 K 个（可按需改成随机）
    k30 = int(max(1, min(args.joint30_size, len(kept_sids))))
    joints_30 = joints_all[:k30]

    # joint_zero: 只写 1 行（对应 zero_sid）
    zero_idx = kept_sids.index(zero_sid)
    joints_zero = joints_all[zero_idx:zero_idx + 1]

    # write joint files
    write_joint_file(out_root / "joint.txt", joints_all)
    write_joint_file(out_root / "joint30.txt", joints_30)
    write_joint_file(out_root / "joint_zero.txt", joints_zero)

    # export RGBA images to out_root/images/<cam>/<sid>.png
    sid_cam_to_out_img: Dict[Tuple[int, str], Path] = {}
    for cam in cams:
        ensure_dir(out_img_root / cam.name)

    for sid in kept_sids:
        for cam in cams:
            cam_dir = args.rgb_dir / cam.name
            rgb_in = find_rgb_path(cam_dir, sid, args.rgb_ext)
            mask_in = find_mask_path(cam_dir, sid, args.mask_suffix, rgb_in.stem)

            out_img = out_img_root / cam.name / f"{sid:06d}.png"
            rgba = load_rgba(rgb_in, mask_in, apply_mask_to_rgb=bool(args.apply_mask_to_rgb))
            rgba.save(out_img)

            sid_cam_to_out_img[(sid, cam.name)] = out_img

    # split train/test by sid
    rng = random.Random(int(args.seed))
    sids_shuf = kept_sids[:]
    rng.shuffle(sids_shuf)
    n_test = int(round(len(sids_shuf) * float(args.test_frac)))
    n_test = min(max(n_test, 0), len(sids_shuf))
    test_sids = sorted(sids_shuf[:n_test])
    train_sids = sorted(sids_shuf[n_test:])

    # keep zero in train (避免你未来想用 test 时出现泄漏/或者 stage1 训练看到了 test)
    if zero_sid in test_sids:
        test_sids.remove(zero_sid)
        train_sids.append(zero_sid)
        train_sids = sorted(train_sids)

    # intrinsics (provider 只会用 frames[0] 的 fov_x/fov_y)
    fov_y = float(cams[0].fovy_rad)
    fov_x = float(fovy_to_fovx(fov_y, w, h))

    # ---------------- write info jsons ----------------
    # mapping: all
    sid_to_joint_id_all = {sid: i for i, sid in enumerate(kept_sids)}
    # mapping: 30d (only first k30)
    sids_30 = kept_sids[:k30]
    sid_to_joint_id_30 = {sid: i for i, sid in enumerate(sids_30)}
    # mapping: 0 (only zero)
    sid_to_joint_id_0 = {zero_sid: 0}

    # all
    info_all_train = build_info_json(
        cams=cams,
        sids=train_sids,
        sid_to_joint_id=sid_to_joint_id_all,
        sid_cam_to_out_img=sid_cam_to_out_img,
        w=w, h=h,
        fov_x=fov_x, fov_y=fov_y,
        relative_paths=bool(args.relative_paths),
    )
    (out_root / "info_all_train.json").write_text(json.dumps(info_all_train, indent=2), encoding="utf-8")

    if test_sids:
        info_all_test = build_info_json(
            cams=cams,
            sids=test_sids,
            sid_to_joint_id=sid_to_joint_id_all,
            sid_cam_to_out_img=sid_cam_to_out_img,
            w=w, h=h,
            fov_x=fov_x, fov_y=fov_y,
            relative_paths=bool(args.relative_paths),
        )
        (out_root / "info_all_test.json").write_text(json.dumps(info_all_test, indent=2), encoding="utf-8")

    # 30d
    # 注意：30d 的 joint_id 必须索引 joint30.txt（长度 k30），所以只写 sids_30 里的样本
    info_30d_train = build_info_json(
        cams=cams,
        sids=[sid for sid in train_sids if sid in sid_to_joint_id_30],
        sid_to_joint_id=sid_to_joint_id_30,
        sid_cam_to_out_img=sid_cam_to_out_img,
        w=w, h=h,
        fov_x=fov_x, fov_y=fov_y,
        relative_paths=bool(args.relative_paths),
    )
    (out_root / "info_30d_train.json").write_text(json.dumps(info_30d_train, indent=2), encoding="utf-8")

    if test_sids:
        info_30d_test = build_info_json(
            cams=cams,
            sids=[sid for sid in test_sids if sid in sid_to_joint_id_30],
            sid_to_joint_id=sid_to_joint_id_30,
            sid_cam_to_out_img=sid_cam_to_out_img,
            w=w, h=h,
            fov_x=fov_x, fov_y=fov_y,
            relative_paths=bool(args.relative_paths),
        )
        (out_root / "info_30d_test.json").write_text(json.dumps(info_30d_test, indent=2), encoding="utf-8")

    # 0 (zero)
    info_0_train = build_info_json(
        cams=cams,
        sids=[zero_sid],
        sid_to_joint_id=sid_to_joint_id_0,
        sid_cam_to_out_img=sid_cam_to_out_img,
        w=w, h=h,
        fov_x=fov_x, fov_y=fov_y,
        relative_paths=bool(args.relative_paths),
    )
    (out_root / "info_0_train.json").write_text(json.dumps(info_0_train, indent=2), encoding="utf-8")

    # optional test for 0 (same as train)
    (out_root / "info_0_test.json").write_text(json.dumps(info_0_train, indent=2), encoding="utf-8")

    # ---------------- report ----------------
    print("\n=== Export Done ===")
    print(f"out_root: {out_root.resolve()}")
    print(f"kept frames(sids): {len(kept_sids)} | dropped_missing={dropped_missing} dropped_smallmask={dropped_smallmask}")
    print(f"cameras: {[c.name for c in cams]}")
    print(f"image size: {w} x {h}")
    print(f"joint dim (D): {D}")
    print(f"joint_out mode: {args.joint_out} (provider 会再 /pi)")
    print(f"zero_sid: {zero_sid:06d}")
    print("\nTrain (不要用 -O；也不要开 --cuda_ray)：")
    print(f"  python main_dnerf.py {out_root}.0   --workspace <ws> --joints_num {D} --fp16 --preload --min_near 0.1 --break_iter 0")
    print(f"  python main_dnerf.py {out_root}.all --workspace <ws> --joints_num {D} --fp16 --preload --min_near 0.1 --break_iter 0")
    print("\n如果相机方向不对（上下/左右翻），尝试在导出时加：--no_flip_yz")


if __name__ == "__main__":
    main()
'''

python make_selfmodel_dnerf_dataset.py \
  --xml tdcr2_no_base.xml \
  --motor_dir 2m_no_base/motor \
  --rgb_dir 2m_no_base/rgb \
  --out_root dnerf/2m_no_base \
  --seed 42 \
  --apply_mask_to_rgb \
  --min_mask_area_frac 0.005 \
  --test_frac 0.1 \
  --joint_out norm_pm1


'''