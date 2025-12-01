from __future__ import annotations
import numpy as np
from pathlib import Path
from .config import NormCfg
from .utils import ensure_dir, print_color, save_ply_ascii

try:
    import h5py
except Exception:
    h5py = None

def _list_shards(root: Path, splits):
    shards = []
    for sp in splits:
        sdir = root / sp
        fs = sorted(p for p in sdir.glob("shard-*.h5"))
        for f in fs:
            shards.append((sp, f))
    return shards

def _compute_center_scale_per_sample(P: np.ndarray, anchor: str = "centroid"):
    if P.size == 0:
        c = np.zeros(3, np.float32); s = np.float32(1.0)
        return c, s
    if anchor == "centroid":
        c = P.mean(axis=0, keepdims=True).astype(np.float32)
        Q = (P.astype(np.float32) - c)
        r = np.linalg.norm(Q, axis=1).max().astype(np.float32)
        s = float(max(r, 1e-6))
        return c.reshape(-1).astype(np.float32), np.float32(s)
    elif anchor == "origin":
        c = np.zeros((1,3), np.float32)
        r = np.linalg.norm(P.astype(np.float32), axis=1).max().astype(np.float32)
        s = float(max(r, 1e-6))
        return c.reshape(-1), np.float32(s)
    else:
        raise ValueError(f"Unknown anchor: {anchor}")

def _compute_global_center_scale(shards, scope: str = "all", anchor: str = "centroid"):
    from tqdm import tqdm as _tqdm
    if anchor == "origin":
        centers = {}
        scales = {}
        keys = set(["ALL"]) if scope == "all" else set(sp for sp, _ in shards)
        for k in keys:
            centers[k] = np.zeros(3, np.float32)
            scales[k] = 1e-6
        for sp, fp in _tqdm(shards, desc="[global-origin] max radius", ncols=120):
            key = sp if scope == "split" else "ALL"
            with h5py.File(fp, "r") as f:
                D = f["data"]; B = D.shape[0]
                for i in range(B):
                    P = D[i].astype(np.float32)
                    r = np.linalg.norm(P, axis=1).max().item()
                    if r > scales[key]:
                        scales[key] = float(r)
        out = { (k if k != "ALL" else "all"): (centers[k], np.float32(scales[k])) for k in centers }
        return out

    sums = {}
    counts = {}
    for sp, fp in _tqdm(shards, desc="[global] pass1 center", ncols=120):
        with h5py.File(fp, "r") as f:
            D = f["data"]
            B = D.shape[0]
            ssum = np.zeros(3, np.float64); cnt = 0
            for i in range(B):
                P = D[i].astype(np.float32)
                ssum += P.mean(axis=0, dtype=np.float64)
                cnt += 1
        key = sp if scope == "split" else "ALL"
        sums[key]   = sums.get(key, 0.0) + ssum
        counts[key] = counts.get(key, 0)  + cnt

    centers = {}
    for k in sums:
        centers[k] = (sums[k] / max(counts[k], 1)).astype(np.float32)

    scales = {k: 1e-6 for k in centers}
    for sp, fp in _tqdm(shards, desc="[global] pass2 radius", ncols=120):
        key = sp if scope == "split" else "ALL"
        C = centers[key].reshape(1,3).astype(np.float32)
        with h5py.File(fp, "r") as f:
            D = f["data"]; B = D.shape[0]
            for i in range(B):
                P = D[i].astype(np.float32)
                r = np.linalg.norm(P - C, axis=1).max().item()
                if r > scales[key]:
                    scales[key] = float(r)
    for k in scales:
        scales[k] = float(max(scales[k], 1e-6))
    out = { (k if k != "ALL" else "all"): (centers[k], np.float32(scales[k])) for k in centers }
    return out

def _save_global_cs(root: Path, cs_map: dict, tag: str):
    import json as _json
    out_npz = root / f"global_norm_{tag}.npz"
    keys = sorted(cs_map.keys())
    centers = np.stack([cs_map[k][0] for k in keys], axis=0).astype(np.float32)
    scales  = np.array([float(cs_map[k][1]) for k in keys], dtype=np.float32)
    np.savez(out_npz, keys=np.array(keys), centers=centers, scales=scales)
    with open(root / f"global_norm_{tag}.json","w") as f:
        _json.dump({k: {"center": cs_map[k][0].tolist(), "scale": float(cs_map[k][1])} for k in keys}, f, indent=2)
    print_color(f"[global] saved center/scale to {out_npz} (and .json)")

def _ensure_datasets(f, count: int, npoints: int, dtype_norm, overwrite=False):
    if "data_norm" in f:
        if overwrite: del f["data_norm"]
        else: raise RuntimeError("data_norm already exists; use --overwrite to replace.")
    if "center" in f:
        if overwrite: del f["center"]
        else: raise RuntimeError("center already exists; use --overwrite to replace.")
    if "scale" in f:
        if overwrite: del f["scale"]
        else: raise RuntimeError("scale already exists; use --overwrite to replace.")
    f.create_dataset("data_norm", shape=(count, npoints, 3), dtype=dtype_norm)
    f.create_dataset("center",    shape=(count, 3), dtype=np.float32)
    f.create_dataset("scale",     shape=(count,),  dtype=np.float32)

def _write_per_sample(fp: Path, dtype_norm, anchor="centroid", overwrite=False):
    from tqdm import tqdm as _tqdm
    with h5py.File(fp, "r+") as f:
        D = f["data"]
        B, N, _ = D.shape
        _ensure_datasets(f, B, N, dtype_norm, overwrite=overwrite)
        Dn = f["data_norm"]; C = f["center"]; S = f["scale"]
        for i in _tqdm(range(B), desc=f"[per-sample:{anchor}] {fp.name}", ncols=120):
            P = D[i].astype(np.float32)
            c, s = _compute_center_scale_per_sample(P, anchor=anchor)
            if anchor == "centroid":
                Pn = (P - c.reshape(1,3)) / s
            else:
                Pn = P / s
            Dn[i] = Pn.astype(dtype_norm)
            C[i]  = c
            S[i]  = s

def _write_global(fp: Path, dtype_norm, global_cs_map, scope_key, anchor="centroid", overwrite=False):
    from tqdm import tqdm as _tqdm
    with h5py.File(fp, "r+") as f:
        D = f["data"]
        B, N, _ = D.shape
        _ensure_datasets(f, B, N, dtype_norm, overwrite=overwrite)
        Dn = f["data_norm"]; C = f["center"]; S = f["scale"]
        Cg, Sg = global_cs_map[scope_key]
        Cg = Cg.astype(np.float32); Sg = float(Sg)
        for i in _tqdm(range(B), desc=f"[global-{scope_key}:{anchor}] {fp.name}", ncols=120):
            P = D[i].astype(np.float32)
            if anchor == "centroid":
                Pn = (P - Cg.reshape(1,3)) / Sg
            else:
                Pn = P / Sg
            Dn[i] = Pn.astype(dtype_norm)
            C[i]  = Cg
            S[i]  = np.float32(Sg)

def _export_random_ply(root: Path, splits: list[str], k: int, out_dir: Path, seed: int = 42):
    rng = np.random.RandomState(seed)
    refs = []
    for sp in splits:
        sdir = root / sp
        for fp in sorted(sdir.glob("shard-*.h5")):
            with h5py.File(fp, "r") as f:
                if "data" not in f or "data_norm" not in f: continue
                B = f["data"].shape[0]
            for i in range(B):
                refs.append((sp, fp, i))
    if not refs:
        print_color("[export_ply] no samples found (ensure data_norm exists).");
        return 0
    k = min(k, len(refs))
    sel = rng.choice(len(refs), size=k, replace=False)
    ensure_dir(out_dir)
    for j, ridx in enumerate(sel):
        sp, fp, i = refs[int(ridx)]
        with h5py.File(fp, "r") as f:
            Pw = f["data"][i].astype(np.float32)
            Pn = f["data_norm"][i].astype(np.float32)
        stem = f"{sp}_{Path(fp).stem}_idx-{i:05d}"
        save_ply_ascii(Pw, Path(out_dir)/f"{stem}_world.ply")
        save_ply_ascii(Pn, Path(out_dir)/f"{stem}_norm.ply")
    print_color(f"📦 [export_ply] wrote {k*2} files to {out_dir}")
    return k

def norm_stage(cfg: NormCfg):
    if h5py is None:
        raise RuntimeError("阶段3需要 h5py 依赖。")
    splits = cfg.splits or ["train","val","test"]
    root = Path(cfg.root)
    work_root = root
    if cfg.no_inplace:
        if not cfg.dst_root:
            raise SystemExit("--no_inplace 需要指定 --dst_root")
        dst_root = ensure_dir(Path(cfg.dst_root))
        import shutil
        for sp in splits:
            ensure_dir(dst_root/sp)
            for fp in sorted((root/sp).glob("shard-*.h5")):
                dst_fp = dst_root/sp/fp.name
                if not dst_fp.exists():
                    shutil.copy2(fp, dst_fp)
        work_root = dst_root

    shards = _list_shards(work_root, splits)
    if not shards:
        raise SystemExit("No shards found under given splits.")
    dtype_norm = np.float32 if cfg.dtype == "float32" else np.float16

    if cfg.mode == "per-sample":
        for sp, fp in shards:
            _write_per_sample(fp, dtype_norm, anchor=cfg.anchor, overwrite=cfg.overwrite)
    else:
        cs_map = _compute_global_center_scale(shards, scope=cfg.scope, anchor=cfg.anchor)
        print_color(f"[global:{cfg.anchor}] center/scale:")
        for k, (c, s) in cs_map.items():
            print_color(f"  - {k}: center={c.tolist()} scale={float(s):.6f}")
        if cfg.dump_global:
            tag = (f"scope-{cfg.scope}_anchor-{cfg.anchor}")
            _save_global_cs(work_root, cs_map, tag)
        for sp, fp in shards:
            key = sp if cfg.scope == "split" else "all"
            _write_global(fp, dtype_norm, cs_map, key, anchor=cfg.anchor, overwrite=cfg.overwrite)
    if cfg.export_ply and cfg.export_ply > 0:
        _export_random_ply(work_root, splits, cfg.export_ply, cfg.export_dir, seed=cfg.export_seed)
    print_color("✅ [normalize] done.")
