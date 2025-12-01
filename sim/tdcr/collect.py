from __future__ import annotations
import os, sys, time, json, math
import numpy as np
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass

from .config import CollectCfg
from .utils import ensure_dir, print_color

# Optional deps
try:
    import mujoco
    from mujoco.renderer import Renderer
except Exception:
    mujoco = None
    Renderer = None
try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import psutil  # optional
except Exception:
    psutil = None


def _detect_actuators(model):
    """
    通用版：**不再要求名字 motor1..**。
    返回：act_ids(np.int32[D])、act_names(list[str])、D。
    顺序 = XML 里 actuator 的顺序（MjModel 本身的顺序）。
    """
    D = int(model.nu)  # num of actuators
    if D <= 0:
        raise RuntimeError("模型不含任何 actuator（无法控制 data.ctrl）。请检查 XML。")
    ids = np.arange(D, dtype=np.int32)
    names = []
    for i in range(D):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
        if nm is None:
            nm = f"actuator{i + 1}"
        names.append(str(nm))
    return ids, names, D


# ---------- math: camera intrinsics & backproject ----------
def _intrinsics(model, res_w, res_h, cid):
    fovy = float(model.cam_fovy[cid])
    fy = res_h / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    fx = fy
    cx, cy = (res_w - 1) / 2.0, (res_h - 1) / 2.0
    return np.float32(fx), np.float32(fy), np.float32(cx), np.float32(cy)


def _cam_to_world_fast(rgb, depth, fx, fy, cx, cy, R, p, depth_max, idx_s):
    z = depth[idx_s].astype(np.float32)
    m = (z > 0) & (z < depth_max)
    if not np.any(m):
        return None, None
    u_sel = idx_s[1][m].astype(np.float32)
    v_sel = idx_s[0][m].astype(np.float32)
    z = z[m]
    x = (u_sel - cx) * z / fx
    y = -(v_sel - cy) * z / fy
    zc = -z
    cam = np.stack((x, y, zc), axis=1)
    world = cam @ R.T + p
    col = (rgb[idx_s][m] / 255.0).astype(np.float32)
    return world, col


def _render_rgbd(renderer: Renderer, data, cid: int):
    renderer.update_scene(data, camera=cid)
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    rgb = renderer.render()
    return rgb, depth


# ---------- sampling ----------
def _generate_controls_discrete(cfg: CollectCfg, model, motor_ids):
    lo = model.actuator_ctrlrange[motor_ids, 0].astype(np.float32)
    hi = model.actuator_ctrlrange[motor_ids, 1].astype(np.float32)
    D = len(motor_ids)
    L = int(cfg.levels_per_motor)
    levels = [np.linspace(lo[i], hi[i], L, dtype=np.float32) for i in range(D)]

    # 组合数估计
    total = L ** D
    # 如果组合数 <= 2e6，就真笛卡尔；否则用随机量化离散（等价于在网格上随机采样）
    if total <= 2_000_000:
        from itertools import product
        grid = np.array(list(product(*levels)), dtype=np.float32)
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(grid, axis=0)
        assert cfg.nsample <= len(grid), "采样数量超过离散组合总数"
        return grid[:cfg.nsample]

    # 随机量化离散：严格去重版
    rng = np.random.default_rng(cfg.seed)
    span = hi - lo
    acc = []
    seen = set()
    while len(acc) < cfg.nsample:
        m = min(8192, cfg.nsample - len(acc))
        x = lo + span * rng.random((m, D), dtype=np.float32)
        # 逐维量化到最近网格点
        for i in range(m):
            for d in range(D):
                # levels[d] 是 1D 数组
                lv = levels[d]
                x[i, d] = lv[np.argmin(np.abs(lv - x[i, d]))]
        # 严格去重
        for i in range(m):
            k = x[i].tobytes()
            if k not in seen:
                seen.add(k)
                acc.append(x[i].copy())
                if len(acc) >= cfg.nsample:
                    break
    return np.stack(acc, axis=0)


def _generate_controls_continuous(cfg: CollectCfg, model, motor_ids):
    rng = np.random.default_rng(cfg.seed)
    lo = model.actuator_ctrlrange[motor_ids, 0].astype(np.float32)
    hi = model.actuator_ctrlrange[motor_ids, 1].astype(np.float32)
    span = hi - lo
    D = len(motor_ids)

    n = cfg.nsample
    block = min(20000, max(1024, n // 10))
    seen = set()
    out = np.zeros((n, D), dtype=np.float32)

    def quantize(v: np.ndarray) -> np.ndarray:
        if cfg.unique_tol is None:
            return v
        q = np.round((v - lo) / cfg.unique_tol).astype(np.int64)
        return lo + q * cfg.unique_tol

    have_min_gap = (cfg.min_gap is not None and cfg.min_gap > 0)
    idx = 0
    tries = 0
    max_tries = 50 * n
    while idx < n and tries < max_tries:
        tries += 1
        m = min(block, n - idx)
        cand = lo + span * rng.random((m, D), dtype=np.float32)
        cand = quantize(cand)

        if have_min_gap:
            keep_mask = np.ones(m, dtype=bool)
            if idx > 0:
                sample_k = min(2048, idx)
                sel = rng.choice(idx, size=sample_k, replace=False)
                base = out[sel]
                chunk = 256
                for s in range(0, m, chunk):
                    e = min(m, s + chunk)
                    cc = cand[s:e][:, None, :]  # (e-s, 1, D)
                    bb = base[None, :, :]  # (1, k, D)
                    d = np.max(np.abs(cc - bb), axis=2)  # (e-s, k)
                    near = (d < cfg.min_gap).any(axis=1)
                    keep_mask[s:e] &= ~near
            cand = cand[keep_mask]
            if len(cand) == 0:
                continue

        kv = [x.tobytes() for x in cand]
        unique_new = [cand[i] for i, k in enumerate(kv) if k not in seen]
        if not unique_new:
            continue
        for x in unique_new:
            seen.add(x.tobytes())
            out[idx] = x
            idx += 1
            if idx >= n:
                break

    if idx < n:
        raise RuntimeError(
            f"连续采样在 {tries} 次尝试后仍未凑齐 {n} 个唯一控制量，"
            f"请降低 min_gap 或 unique_tol，或减少 nsample。"
        )
    return out


# ---------- stability (early-stop) ----------
def _relax_to_stable(model, data, max_steps, vel_eps, qpos_eps, win, zero_vel=True):
    if zero_vel:
        data.qvel[:] = 0
        data.qacc[:] = 0
        data.act[:] = 0
    mujoco.mj_forward(model, data)
    prev = data.qpos.copy()
    ok = 0
    vmax = 0.0;
    dq = 0.0
    for s in range(max_steps):
        mujoco.mj_step(model, data)
        vmax = float(np.max(np.abs(data.qvel)))
        dq = float(np.max(np.abs(data.qpos - prev)))
        prev[:] = data.qpos
        if vmax < vel_eps and dq < qpos_eps:
            ok += 1
            if ok >= win:
                return s + 1, True, vmax, dq
        else:
            ok = 0
    return max_steps, False, vmax, dq


# ---------- GL backend selection ----------
def _pick_gl_backend(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if sys.platform.startswith("win"):
        return "glfw"
    headless = ("DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ)
    return "egl" if headless else "glfw"


def _make_renderer_robust(model, w, h, prefer="auto", egl_quiet=True):
    if egl_quiet:
        os.environ.setdefault("LIBEGL_DEBUG", "fatal")
        os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
    tried = []

    def _try(b):
        os.environ["MUJOCO_GL"] = b
        try:
            r = Renderer(model, width=w, height=h)
            print_color(f"[collect] Renderer backend: {b}")
            return r
        except Exception as e:
            tried.append((b, str(e)))
            return None

    p = _pick_gl_backend(prefer)
    order = []
    if p == "egl":
        order = ["egl", "osmesa", "glfw"]
    elif p == "glfw":
        order = ["glfw", "egl", "osmesa"]
    else:
        order = [p, "egl", "osmesa", "glfw"]
    for b in order:
        r = _try(b)
        if r is not None:
            return r
    raise RuntimeError("Renderer init failed: " + " | ".join(f"{b}:{m}" for b, m in tried))


def _auto_worker_count(wish, res_w, res_h, ncam):
    cpu = max(1, os.cpu_count() or 4)
    if wish and wish > 0:
        return max(1, min(wish, cpu))
    per_worker_gb = 0.35 + 0.008 * (res_w * res_h / 1e6) * max(1, ncam)
    avail_gb = None
    try:
        if psutil is not None:
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        avail_gb = None
    if avail_gb is None:
        return max(1, cpu // 2)
    return max(1, min(cpu, int(0.75 * avail_gb / max(0.1, per_worker_gb))))


# ---------- prewrite motors ----------
def _prewrite_all_motors(ctrls: np.ndarray, out_dir: Path, start: int, act_names: list[str]):
    ensure_dir(out_dir)
    kv = [x.tobytes() for x in ctrls]
    if len(kv) != len(set(kv)):
        raise AssertionError("预生成的 motors 出现重复，请调小 --min_gap 或调整采样参数")
    D = ctrls.shape[1]
    for i, ctrl in enumerate(ctrls, start):
        k = f"{i:06d}.json"
        p = out_dir / k
        if not p.exists():
            # 新格式，显式包含按 XML 顺序的控制量数组，附带名字方便追踪
            obj = {"ctrl": [float(v) for v in ctrl.tolist()], "actuator_names": act_names}
            with open(p, "w") as f:
                json.dump(obj, f, indent=4, ensure_ascii=False)


# ---------- worker ----------
def _collect_worker(args):
    cfg_dict, indices, position = args
    from tqdm import tqdm as _tqdm

    xml = cfg_dict["xml"]
    res_w = cfg_dict["res_w"]
    res_h = cfg_dict["res_h"]
    depth_max = cfg_dict["depth_max"]
    stride = cfg_dict["stride"]
    seed = cfg_dict["seed"]
    out_pcd_dir = Path(cfg_dict["out_pcd_dir"])
    out_json_dir = Path(cfg_dict["out_json_dir"])
    backend = cfg_dict["backend"]
    egl_quiet = cfg_dict["egl_quiet"]
    sim_steps = cfg_dict["sim_steps"]
    use_relax = cfg_dict["use_relax"]
    relax_max = cfg_dict["relax_max_steps"]
    vel_eps = cfg_dict["stable_vel_eps"]
    qpos_eps = cfg_dict["stable_qpos_eps"]
    win = cfg_dict["stable_win"]
    zero_vel = cfg_dict["zero_vel_each_ctrl"]
    resume = cfg_dict["resume"]

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = _make_renderer_robust(model, res_w, res_h, prefer=backend, egl_quiet=egl_quiet)

    motor_ids, motor_names, D = _detect_actuators(model)
    cam_ids = list(range(model.ncam))

    U_full, V_full = np.meshgrid(np.arange(res_w, dtype=np.float32),
                                 np.arange(res_h, dtype=np.float32))
    U_s = U_full[::stride, ::stride].ravel().astype(np.intp)
    V_s = V_full[::stride, ::stride].ravel().astype(np.intp)
    idx_s = (V_s, U_s)

    rng = np.random.default_rng(seed + position * 1_000_003)
    bar = _tqdm(indices, position=position, ncols=120, desc=f"worker-{position}", leave=True)
    for k in bar:
        pcd_path = out_pcd_dir / f"{k:06d}.ply"
        if resume and pcd_path.exists():
            continue
        with open(out_json_dir / f"{k:06d}.json", "r") as f:
            obj = json.load(f)
        if isinstance(obj, (list, tuple)):
            ctrl = np.array(obj, dtype=np.float32).reshape(-1)
        elif isinstance(obj, dict):
            if "ctrl" in obj:
                ctrl = np.array(obj["ctrl"], dtype=np.float32).reshape(-1)
            else:
                # 兼容旧格式: motor1..motorD
                vals = []
                for j in range(1, D + 1):
                    key = f"motor{j}"
                    if key not in obj:
                        raise KeyError(f"{k:06d}.json 缺少键 {key}")
                    vals.append(obj[key])
                ctrl = np.array(vals, dtype=np.float32).reshape(-1)
        else:
            raise ValueError("不支持的 JSON motor 格式")
        assert len(ctrl) == D, f"{k:06d}.json: 期望 {D} 维，实际 {len(ctrl)}"

        data.ctrl[motor_ids] = ctrl

        if use_relax:
            _relax_to_stable(model, data, relax_max, vel_eps, qpos_eps, win, zero_vel=zero_vel)
        else:
            for _ in range(sim_steps):
                mujoco.mj_step(model, data)

        pts_all, col_all = [], []
        for cid in cam_ids:
            rgb, d = _render_rgbd(renderer, data, cid)
            fx, fy, cx, cy = _intrinsics(model, res_w, res_h, cid)
            R = data.cam_xmat[cid].reshape(3, 3).astype(np.float32)
            p = data.cam_xpos[cid].astype(np.float32)
            pts, col = _cam_to_world_fast(rgb, d, fx, fy, cx, cy, R, p, depth_max, idx_s)
            if pts is not None:
                pts_all.append(pts);
                col_all.append(col)
        if pts_all:
            pts = np.concatenate(pts_all, axis=0)
            col = np.concatenate(col_all, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(col)
            o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)
    renderer.close()


# ---------- public entry ----------
def collect_stage(cfg: CollectCfg):
    if mujoco is None or Renderer is None or o3d is None:
        raise RuntimeError("阶段1需要 mujoco、mujoco.renderer.Renderer 与 open3d 依赖。")
    ensure_dir(cfg.out_pcd_dir)
    ensure_dir(cfg.out_json_dir)

    # 预载模型，用来获知相机数量、控制范围等
    model_probe = mujoco.MjModel.from_xml_path(cfg.xml)
    ncam = int(model_probe.ncam)

    # === 统一计算“最终并发数” nworkers（支持 --workers<=0 的自动模式） ===
    if cfg.workers <= 0:
        nworkers = _auto_worker_count(cfg.workers, cfg.res_w, cfg.res_h, ncam)
    else:
        nworkers = int(cfg.workers)
    # 受样本数上限约束，且至少为 1
    nworkers = max(1, min(int(nworkers), int(cfg.nsample)))
    how = "auto" if cfg.workers <= 0 else "explicit"
    print_color(f"[collect] multiprocessing: {'ON' if nworkers > 1 else 'OFF'} "
                f"(workers={nworkers}, mode={how}, ctx={cfg.ctx})")

    # 生成电机控制量（全局唯一），并在磁盘预写 JSON 以保障并行安全
    motor_ids_probe, motor_names_probe, D = _detect_actuators(model_probe)
    print_color(f"[collect] detected motors: D={D} (segments={D // 3})")
    if cfg.sampling == "discrete":
        all_ctrls = _generate_controls_discrete(cfg, model_probe, motor_ids_probe)
    else:
        all_ctrls = _generate_controls_continuous(cfg, model_probe, motor_ids_probe)

    if cfg.resume and len(list(Path(cfg.out_pcd_dir).glob("*.ply"))) > 0:
        print_color("[collect] resume is on: 将跳过已存在 .ply 的样本。")

    _prewrite_all_motors(all_ctrls, cfg.out_json_dir, cfg.start_index, motor_names_probe)

    # ===== 串行分支 =====
    if nworkers == 1:
        data = mujoco.MjData(model_probe)
        mujoco.mj_forward(model_probe, data)
        renderer = _make_renderer_robust(
            model_probe, cfg.res_w, cfg.res_h,
            prefer=cfg.backend, egl_quiet=cfg.egl_quiet
        )

        motor_ids = motor_ids_probe
        cam_ids = list(range(model_probe.ncam))

        U_full, V_full = np.meshgrid(
            np.arange(cfg.res_w, dtype=np.float32),
            np.arange(cfg.res_h, dtype=np.float32)
        )
        U_s = U_full[::cfg.stride, ::cfg.stride].ravel().astype(np.intp)
        V_s = V_full[::cfg.stride, ::cfg.stride].ravel().astype(np.intp)
        idx_s = (V_s, U_s)

        from tqdm import tqdm as _tqdm
        for i in _tqdm(range(cfg.nsample), ncols=100, desc="collect"):
            k = cfg.start_index + i
            pcd_path = Path(cfg.out_pcd_dir) / f"{k:06d}.ply"
            if cfg.resume and pcd_path.exists():
                continue

            ctrl = all_ctrls[i]
            data.ctrl[motor_ids] = ctrl

            # 稳定-早停 或 固定步
            if cfg.relax_max_steps and cfg.stable_win:
                _relax_to_stable(
                    model_probe, data, cfg.relax_max_steps,
                    cfg.stable_vel_eps, cfg.stable_qpos_eps, cfg.stable_win,
                    zero_vel=cfg.zero_vel_each_ctrl
                )
            else:
                for _ in range(cfg.sim_steps):
                    mujoco.mj_step(model_probe, data)

            # 渲染→拼点云→保存
            pts_all, col_all = [], []
            for cid in cam_ids:
                rgb, d = _render_rgbd(renderer, data, cid)
                fx, fy, cx, cy = _intrinsics(model_probe, cfg.res_w, cfg.res_h, cid)
                R = data.cam_xmat[cid].reshape(3, 3).astype(np.float32)
                p = data.cam_xpos[cid].astype(np.float32)
                pts, col = _cam_to_world_fast(rgb, d, fx, fy, cx, cy, R, p, cfg.depth_max, idx_s)
                if pts is not None:
                    pts_all.append(pts)
                    col_all.append(col)
            if pts_all:
                pts = np.concatenate(pts_all, axis=0)
                col = np.concatenate(col_all, axis=0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(col)
                o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)

        renderer.close()
        print_color("✅ [collect] done.")
        return

    # ===== 并行分支 =====
    all_indices = list(range(cfg.start_index, cfg.start_index + cfg.nsample))
    splits = np.array_split(all_indices, nworkers)
    base_cfg = dict(
        xml=cfg.xml, res_w=cfg.res_w, res_h=cfg.res_h, depth_max=cfg.depth_max, stride=cfg.stride,
        seed=cfg.seed, out_pcd_dir=str(cfg.out_pcd_dir), out_json_dir=str(cfg.out_json_dir),
        backend=cfg.backend, egl_quiet=cfg.egl_quiet, sim_steps=cfg.sim_steps,
        use_relax=bool(cfg.relax_max_steps and cfg.stable_win),
        relax_max_steps=cfg.relax_max_steps, stable_vel_eps=cfg.stable_vel_eps,
        stable_qpos_eps=cfg.stable_qpos_eps, stable_win=cfg.stable_win,
        zero_vel_each_ctrl=cfg.zero_vel_each_ctrl, resume=cfg.resume
    )
    ctx = mp.get_context(cfg.ctx if cfg.ctx in ("spawn", "fork") else "spawn")
    tasks = [(base_cfg, list(map(int, arr)), rank) for rank, arr in enumerate(splits)]
    with ctx.Pool(processes=nworkers) as pool:
        for _ in pool.imap_unordered(_collect_worker, tasks, chunksize=1):
            pass
    print_color("✅ [collect] done (parallel).")

