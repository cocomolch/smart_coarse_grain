#!/usr/bin/env python3
"""
3D Nematic Order Parameter + Disclination Pipeline
====================================================
Processes a multi-page TIFF volume and computes:

  ── 3D ──────────────────────────────────────────────────────────────────────
  • Structure tensor (GPU)  → raw directors (vx, vy, vz), angles (θ, φ)
  • Q-tensor coarse-graining (GPU) → CG directors (nx_cg, ny_cg, nz_cg),
    scalar order parameter S, full Q-tensor components (Qxx_cg … Qyz_cg)
  • Disclination detection via winding-number per orthogonal slice (GPU)
    → charges3D, per-plane maps (winding_xy / xz / yz)
  • Omega field (plane normal of each detected disclination)
  • Beta angle (angle between disclination tangent and omega)
  • Skeletonised disclination lines

  ── 2D (every XY slice, or strided) ─────────────────────────────────────────
  • 2D structure tensor (GPU) per slice → raw 2D directors
  • 2D Q-tensor CG → CG directors + scalar order S_2d
  • Winding-number defect maps (continuous + cleaned ±0.5 map)

Saves two pairs of output files:
  <OUTPUT_DIR>/<EXPERIMENT_NAME>_3d.npz   / .mat
  <OUTPUT_DIR>/<EXPERIMENT_NAME>_2d_xy.npz / .mat

Usage:
    Edit the CONFIG block below, then:
        python analyze_volume_3d.py
    Or pass a config JSON:
        python analyze_volume_3d.py --config my_cfg.json
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import torch as th
import torch.nn.functional as F
import tifffile
import scipy.io as sio
from scipy.ndimage import label

# ── import utils ─────────────────────────────────────────────────────────────

import skimage.morphology as _skmorph
if not hasattr(_skmorph, "skeletonize_3d"):
    _skmorph.skeletonize_3d = _skmorph.skeletonize

sys.path.insert(0, os.path.dirname('C:/Users/pgotthe/Documents/WSINematics/smart_coarse_grain/smart_coarse_grain/'))
import structure_tensor_utils as _stu         
from structure_tensor_utils import (
    # 3D
    smart_structure_tensor_3d_gpu,
    disclination_detection_3d,
    compute_omega_and_beta,
    # helpers exposed by the module
    _gaussian_kernel_3d_gpu,
    _eig_symmetric_3x3,
    # 2D
    smart_structure_tensor_gpu,
    get2DQTensor_gpu,
    defect_analysis,
    _get3DQTensor_gpu_fast,
    zapotocky_plaquette_3d,
    _gradient_disclination_3d
)

# ── NaN-safe, chunked replacement for _eig_symmetric_3x3 ─────────────────────
_EIG_CHUNK = 1_000_000   # lower if you still get OOM

def _eig_symmetric_3x3_safe(Sxx, Syy, Szz, Sxy, Sxz, Syz):
    orig_shape = Sxx.shape
    N = Sxx.numel()

    M = th.zeros(N, 3, 3, device=Sxx.device, dtype=Sxx.dtype)
    for comp, (r, c) in zip(
        [Sxx.reshape(-1), Syy.reshape(-1), Szz.reshape(-1),
         Sxy.reshape(-1), Sxz.reshape(-1), Syz.reshape(-1)],
        [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    ):
        M[:, r, c] = comp
        if r != c:
            M[:, c, r] = comp

    nan_mask = ~th.isfinite(M).all(dim=-1).all(dim=-1)
    n_nan = int(nan_mask.sum())
    if n_nan:
        print(f"  [eig patch] {n_nan}/{N} voxels had NaN/Inf → zeroed before eigh")
    M[nan_mask] = 0.0

    all_vals = th.empty(N, 3, device=Sxx.device, dtype=Sxx.dtype)
    all_vecs = th.empty(N, 3, 3, device=Sxx.device, dtype=Sxx.dtype)
    for start in range(0, N, _EIG_CHUNK):
        end = min(start + _EIG_CHUNK, N)
        v, e = th.linalg.eigh(M[start:end])
        all_vals[start:end] = v
        all_vecs[start:end] = e

    val_min = all_vals[:, 0].reshape(orig_shape)
    vx = all_vecs[:, 0, 0].reshape(orig_shape)
    vy = all_vecs[:, 1, 0].reshape(orig_shape)
    vz = all_vecs[:, 2, 0].reshape(orig_shape)
    return val_min, vx, vy, vz

_stu._eig_symmetric_3x3 = _eig_symmetric_3x3_safe   # patch both 3D functions
def _zapotocky_chunked(nx, ny, nz, slab_size=10):
    """
    Memory-safe wrapper for zapotocky_plaquette_3d.
    Processes Z-slabs of thickness slab_size to avoid allocating
    (D, H, W, 3) corner arrays all at once.
    Slab overlap of 1 slice ensures plaquettes at slab boundaries are covered.
    """
    from structure_tensor_utils import zapotocky_plaquette_3d
    D, H, W = nx.shape
    charges3D  = np.zeros((D, H, W), dtype=np.float32)
    winding_xy = np.zeros((D, H, W), dtype=np.float32)
    winding_xz = np.zeros((D, H, W), dtype=np.float32)
    winding_yz = np.zeros((D, H, W), dtype=np.float32)

    for z0 in range(0, D, slab_size):
        z1 = min(z0 + slab_size + 1, D)   # +1 overlap so boundary plaquettes are included
        print(f"  Zapotocky slab z={z0}:{z1}  ", end="\r")
        c, xy, yz, zx = zapotocky_plaquette_3d(
            nx[z0:z1], ny[z0:z1], nz[z0:z1]
        )
        # write only the non-overlap portion back (except last slab)
        z_out = slice(z0, z1 - 1 if z1 < D else z1)
        charges3D[z_out]  = c [:z1 - z0 - (1 if z1 < D else 0)]
        winding_xy[z_out] = xy[:z1 - z0 - (1 if z1 < D else 0)]
        winding_xz[z_out] = zx[:z1 - z0 - (1 if z1 < D else 0)]
        winding_yz[z_out] = yz[:z1 - z0 - (1 if z1 < D else 0)]

    print()
    return charges3D, winding_xy, winding_xz, winding_yz
# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← edit this block; everything else runs automatically
# ═══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    # ── I/O ──────────────────────────────────────────────────────────────────
    
    #takes not the 3D tiles from director method, but load to fiji and then merge color channels and save as tiff
    INPUT_TIFF        = r"C:\Users\pgotthe\Documents\WSINematics\tiles3D_Fibrosarkoma\tile_x002_y003_wQualityControl\Merged.tif",   # multi-page TIFF (Z, Y, X)
    OUTPUT_DIR        = r"C:\Users\pgotthe\Documents\WSINematics\tiles3D_Fibrosarkoma\tile_x002_y003_wQualityControl",             # created automatically
    EXPERIMENT_NAME   = "nematics3DResults_zapotockyMethod_06Thresh_structuretensor8mum_Qtensor20mum_May4th",                   # prefix for saved files

    # ── Physical calibration ─────────────────────────────────────────────────
    MUM_PER_PX_XY     = 2,    # lateral pixel size  [µm/px]
    MUM_PER_PX_Z      = 2,    # axial   pixel size  [µm/slice]

    # ── 3D structure tensor ───────────────────────────────────────────────────
    CGL_3D_mum        = 8,     # coarse-graining kernel size [µm] — will be converted to voxels using MUM_PER_PX_XY
    CG_METHOD_3D      = "gaussian",  # "gaussian" | "mean"

    # ── 3D Q-tensor coarse-graining ───────────────────────────────────────────
    Q_SCALE_3D_UM     = 20,    # Gaussian σ for Q-tensor averaging [µm]
    EIG_CHUNK_3D = 1_000_000,   # chunk size GPU


    # ── 3D disclination detection ─────────────────────────────────────────────
    RING_RADIUS_3D    = 3,      # ring-kernel radius for winding-number [voxels]
    WINDING_THRESH    = 0.1,    # |winding| threshold to flag a defect voxel
    MIN_LINE_LENGTH   = 1,     # discard skeleton components shorter than this
    DISC_METHOD = "zapotocky",  # "winding" | "zapotocky" | "gradient"

    GRAD_THRESH       = 0.6,   # |∇n| threshold for disclination core — tune between 0.05–0.4
    GRAD_MIN_VOXELS   = 20,     # discard connected components smaller than this
    # ── 2D XY-slice analysis ──────────────────────────────────────────────────
    CGL_2D            = 4,      # coarse-graining kernel size [pixels]
    DOWNSAMPLE_2D     = 2,
    Q_SCALE_2D_UM     = 20,    # Gaussian σ for 2D Q-tensor [µm]
    RING_RADIUS_2D    = 3,      # winding-number ring kernel [pixels]
    SLICE_STRIDE      = 1,      # 1 = every slice, N = every Nth slice

    # ── Hardware ──────────────────────────────────────────────────────────────
    DEVICE            = "cuda",  # "cuda" | "cuda:0" | "cpu"
)



CFG["CGL_3D"] = CFG["CGL_3D_mum"] / CFG["MUM_PER_PX_XY"]
CFG["DOWNSAMPLE_3D"] = int(CFG["CGL_3D"] / 1.5) # structure tensor downsampling for speed (vs. resolution); typically half the CG kernel size
CFG["DOWNSAMPLE_Q_3D"] = int((CFG["Q_SCALE_3D_UM"] / CFG["MUM_PER_PX_XY"]) / 3) # internal downsampling for Q-tensor CG (speed vs. resolution)
CFG["DISC_DOWNSAMPLE"] = int(CFG["Q_SCALE_3D_UM"] // CFG["MUM_PER_PX_XY"] // 2)   # downsample CG directors before disclination detection (speed vs. resolution)

# ═══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ═━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_tiff(path: str) -> np.ndarray:
    """Load a multi-page TIFF as float32 (Z, Y, X)."""
    vol = tifffile.imread(path)
    if vol.ndim == 2:
        vol = vol[np.newaxis]          # single slice → (1, Y, X)
    elif vol.ndim == 4:
        # (Z, C, Y, X) or (C, Z, Y, X) — take first channel
        if vol.shape[1] < vol.shape[0]:
            vol = vol[:, 0]            # (Z, C, Y, X) → (Z, Y, X)
        else:
            vol = vol[0]              # (C, Z, Y, X) → (Z, Y, X)
    assert vol.ndim == 3, f"Expected 3-D volume after loading, got shape {vol.shape}"
    vol = vol.astype(np.float32)
    # normalise to [0, 1]
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    print(f"  Loaded volume: {vol.shape}  (D={vol.shape[0]}, H={vol.shape[1]}, W={vol.shape[2]})")
    return vol


import h5py
import threading

def _save_pair(npz_path: str, mat_path: str, arrays: dict, meta: dict):
    """Save dict of arrays to .npz (uncompressed) and .mat (HDF5 v7.3) in parallel."""

    def _save_npz():
        np.savez(npz_path, **arrays,
                 **{f"meta_{k}": np.array(v) for k, v in meta.items()})
        size_mb = os.path.getsize(npz_path) / 1e6
        print(f"  Saved: {npz_path}  ({size_mb:.0f} MB)")

    def _save_mat():
        with h5py.File(mat_path, "w") as f:
            # mark as MATLAB v7.3
            f.attrs["MATLAB_class"] = np.bytes_("double")
            grp = f.create_group("data")
            for k, v in arrays.items():
                safe_k = k.lstrip("0123456789").replace("-", "_")
                arr = np.array(v)
                # h5py is fastest with C-contiguous float32
                grp.create_dataset(safe_k, data=np.ascontiguousarray(arr),
                                   compression=None)
            meta_grp = f.create_group("meta")
            for k, v in meta.items():
                meta_grp.create_dataset(k, data=np.array(v))
        size_mb = os.path.getsize(mat_path) / 1e6
        print(f"  Saved: {mat_path}  ({size_mb:.0f} MB)")

    t1 = threading.Thread(target=_save_npz)
    t2 = threading.Thread(target=_save_mat)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


# ═══════════════════════════════════════════════════════════════════════════════
#  3D pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_3d_pipeline(vol: np.ndarray, cfg: dict) -> dict:
    """
    Full 3-D analysis.

    Returns
    -------
    dict with keys (all np.ndarray unless noted):
      vx, vy, vz              raw structure-tensor directors       (D,H,W)
      theta_3d, phi_3d        polar / azimuthal angle [rad]        (D,H,W)
      nx_cg, ny_cg, nz_cg    Q-tensor CG directors                (D,H,W)
      S_3d                   scalar order parameter  [0,1]         (D,H,W)
      Qxx_cg … Qyz_cg        symmetric Q-tensor components        (D,H,W)
      charges3D              total winding number per voxel        (D,H,W)
      winding_xy/xz/yz       per-plane raw winding maps            (D,H,W)
      omega_3d               plane-normal of detected disclination (D,H,W,3)
      beta_3d                tangent–omega angle [rad]             (D,H,W)
      skel_3d                skeletonised disclination lines  bool (D,H,W)
    """
    device = cfg["DEVICE"]
    print("\n── 3D structure tensor (GPU) ─────────────────────────────────────────")
    t0 = time.time()

    theta_t, phi_t, vx_t, vy_t, vz_t = smart_structure_tensor_3d_gpu(
        vol,
        coarse_grain_average   = cfg["CG_METHOD_3D"],
        coarse_graining_length = cfg["CGL_3D"],
        downsample             = cfg["DOWNSAMPLE_3D"],
        interpolation_method   = "trilinear",
        device                 = device,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # move to CPU numpy
    vx    = vx_t.squeeze().cpu().numpy()
    vy    = vy_t.squeeze().cpu().numpy()
    vz    = vz_t.squeeze().cpu().numpy()
    theta_3d = theta_t.squeeze().cpu().numpy()
    phi_3d   = phi_t.squeeze().cpu().numpy()

    # ── 3D Q-tensor ──────────────────────────────────────────────────────────
    print("\n── 3D Q-tensor coarse-graining (GPU) ────────────────────────────────")
    t0 = time.time()

    nx_cg_t, ny_cg_t, nz_cg_t, S_t, \
    Qxx_cg_t, Qyy_cg_t, Qzz_cg_t, \
    Qxy_cg_t, Qxz_cg_t, Qyz_cg_t =_get3DQTensor_gpu_fast(vx_t, vy_t, vz_t,
        Q_average_scale = cfg["Q_SCALE_3D_UM"],
        mum_per_px      = cfg["MUM_PER_PX_XY"],
        _DOWNSAMPLE_Q   = cfg["DOWNSAMPLE_Q_3D"],   # internal downsampling for Q-tensor CG (speed vs. resolution)
        _EIG_CHUNK      = cfg["EIG_CHUNK_3D"],
        device          = device,)


    nx_cg = nx_cg_t.squeeze().cpu().numpy()
    ny_cg = ny_cg_t.squeeze().cpu().numpy()
    nz_cg = nz_cg_t.squeeze().cpu().numpy()
    S_3d  = S_t.squeeze().cpu().numpy()

    print(f"  Done in {time.time()-t0:.1f}s  |  S mean={S_3d.mean():.3f}  max={S_3d.max():.3f}")


    # free GPU memory
    del vx_t, vy_t, vz_t, nx_cg_t, ny_cg_t, nz_cg_t, S_t
    if "cuda" in device:
        th.cuda.empty_cache()

    # ── 3D disclination detection ─────────────────────────────────────────────
    print(f"\n── 3D disclination detection ({cfg['DISC_METHOD']}) ──────────────────")
    t0 = time.time()

    # initialise outputs that not all methods fill
    loop_mask = np.zeros(nx_cg.shape, dtype=np.uint8)
    genus_map = np.zeros(nx_cg.shape, dtype=np.int32)
    winding_xy   = np.zeros(nx_cg.shape, dtype=np.float32)
    winding_xz   = np.zeros(nx_cg.shape, dtype=np.float32)
    winding_yz   = np.zeros(nx_cg.shape, dtype=np.float32)
    omega_raw_up = np.zeros((*nx_cg.shape, 3), dtype=np.float32)

    if cfg["DISC_METHOD"] == "gradient":
        charges3D, loop_mask, genus_map = _gradient_disclination_3d(
            nx_cg, ny_cg, nz_cg,
            threshold  = cfg["GRAD_THRESH"],
            min_voxels = cfg["GRAD_MIN_VOXELS"],
        )

    elif cfg["DISC_METHOD"] == "zapotocky":
        charges3D, winding_xy, winding_xz, winding_yz = _zapotocky_chunked(
            nx_cg, ny_cg, nz_cg, slab_size=10)
        winding_xy   = np.asarray(winding_xy, dtype=np.float32)
        winding_xz   = np.asarray(winding_xz, dtype=np.float32)
        winding_yz   = np.asarray(winding_yz, dtype=np.float32)
        omega_raw_up[winding_xy > 0, 2] = 1.0
        omega_raw_up[winding_xz > 0, 1] = 1.0
        omega_raw_up[winding_yz > 0, 0] = 1.0

    else:  # "winding"
        ds = cfg["DISC_DOWNSAMPLE"]
        print(f"  Director shape: {nx_cg.shape}  →  downsampling by {ds}")
        if ds > 1:
            def _ds3d(arr):
                t = th.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                out = F.interpolate(t, scale_factor=1/ds, mode='trilinear', align_corners=False)
                return out.squeeze().numpy()
            nx_disc = _ds3d(nx_cg);  ny_disc = _ds3d(ny_cg);  nz_disc = _ds3d(nz_cg)
        else:
            nx_disc, ny_disc, nz_disc = nx_cg, ny_cg, nz_cg
        print(f"  Disclination input shape: {nx_disc.shape}")
        charges3D_ds, winding_xy_ds, winding_xz_ds, winding_yz_ds, omega_raw = disclination_detection_3d(
            nx_disc, ny_disc, nz_disc,
            radius          = cfg["RING_RADIUS_3D"],
            winding_thresh  = cfg["WINDING_THRESH"],
            min_line_length = cfg["MIN_LINE_LENGTH"],
            device          = device,
        )
        native = nx_cg.shape
        if ds > 1:
            def _us3d(arr):
                t = th.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                out = F.interpolate(t, size=native, mode='trilinear', align_corners=False)
                return out.squeeze().numpy()
            charges3D  = _us3d(charges3D_ds)
            winding_xy = _us3d(winding_xy_ds)
            winding_xz = _us3d(winding_xz_ds)
            winding_yz = _us3d(winding_yz_ds)
            omega_raw_up = np.stack([_us3d(omega_raw[..., i]) for i in range(3)], axis=-1)
        else:
            charges3D  = charges3D_ds
            winding_xy = winding_xy_ds
            winding_xz = winding_xz_ds
            winding_yz = winding_yz_ds
            omega_raw_up = omega_raw

    n_defect_vox = int((charges3D > 0).sum())
    print(f"  Done in {time.time()-t0:.1f}s  |  defect voxels: {n_defect_vox}")
    # Use np.asarray instead of .astype to avoid copying arrays that are
    # already float32 — each .astype() on a (150,2048,2048) float32 array
    # allocates an extra 2.34 GiB even when no conversion is needed.
    def _f32(a):
        return np.asarray(a, dtype=np.float32)

    return dict(
        # raw structure tensor
        vx=_f32(vx),
        vy=_f32(vy),
        vz=_f32(vz),
        #theta_3d=_f32(theta_3d),
        #phi_3d=_f32(phi_3d),
        # Q-tensor CG
        nx_cg=_f32(nx_cg),
        ny_cg=_f32(ny_cg),
        nz_cg=_f32(nz_cg),
        S_3d=_f32(S_3d),
        # disclination
        charges3D=_f32(charges3D),
        winding_xy=_f32(winding_xy),
        winding_xz=_f32(winding_xz),
        winding_yz=_f32(winding_yz),
        omega_3d=_f32(omega_raw_up),
        loop_mask  = np.asarray(loop_mask, dtype=np.uint8),
        genus_map  = np.asarray(genus_map, dtype=np.int32),
        #beta_3d=_f32(beta_3d),
        #skel_3d=np.asarray(skel_3d, dtype=np.uint8),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  2D XY-slice pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_2d_xy_pipeline(vol: np.ndarray, cfg: dict) -> dict:
    """
    Per-XY-slice 2-D analysis (independent from 3D structure tensor).

    Returns
    -------
    dict with keys (shape N_slices × H × W):
      nx_2d, ny_2d           raw 2D structure-tensor directors
      nx_cg_2d, ny_cg_2d    Q-tensor CG directors
      S_2d                   scalar order parameter
      winding_2d             raw winding-number map (continuous)
      winding_clean_2d       cleaned ±0.5 defect map
      slice_indices          which Z-indices were analysed
    """
    device = cfg["DEVICE"]
    D, H, W = vol.shape
    stride  = cfg["SLICE_STRIDE"]
    z_indices = list(range(0, D, stride))
    N = len(z_indices)

    print(f"\n── 2D XY-slice analysis: {N} slices (stride={stride}) ───────────────")

    # pre-allocate output arrays
    nx_2d           = np.zeros((N, H, W), dtype=np.float32)
    ny_2d           = np.zeros((N, H, W), dtype=np.float32)
    nx_cg_2d        = np.zeros((N, H, W), dtype=np.float32)
    ny_cg_2d        = np.zeros((N, H, W), dtype=np.float32)
    S_2d            = np.zeros((N, H, W), dtype=np.float32)
    winding_2d      = np.zeros((N, H, W), dtype=np.float32)
    winding_clean_2d = np.zeros((N, H, W), dtype=np.float32)

    t_total = time.time()
    for out_idx, z in enumerate(z_indices):
        if out_idx % max(1, N // 10) == 0:
            elapsed = time.time() - t_total
            eta = elapsed / max(out_idx, 1) * (N - out_idx)
            print(f"  Slice {out_idx+1}/{N}  (z={z})  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        # ── 2D structure tensor ──────────────────────────────────────────────
        slc = vol[z]                                                  # (H, W)
        img_t = th.from_numpy(slc).unsqueeze(0).unsqueeze(0)         # [1,1,H,W]

        nx_t, ny_t = smart_structure_tensor_gpu(
            img_t,
            coarseGrainAverage  = cfg["CG_METHOD_3D"],   # reuse 3D method flag
            coarseGrainingLength= cfg["CGL_2D"],
            downsample          = cfg["DOWNSAMPLE_2D"],
            device              = device,
        )
        nx_2d[out_idx] = nx_t.squeeze().cpu().numpy()
        ny_2d[out_idx] = ny_t.squeeze().cpu().numpy()

        # ── 2D Q-tensor CG ───────────────────────────────────────────────────
        nx_cg_t, ny_cg_t, S_t = get2DQTensor_gpu(
            nx_t, ny_t,
            QtensorAverageScale = cfg["Q_SCALE_2D_UM"],
            mum_per_px_2D       = cfg["MUM_PER_PX_XY"],
        )
        nx_cg_2d[out_idx] = nx_cg_t.squeeze().cpu().numpy()
        ny_cg_2d[out_idx] = ny_cg_t.squeeze().cpu().numpy()
        S_2d[out_idx]     = S_t.squeeze().cpu().numpy()

        # ── 2D defect detection (CPU, numpy) ─────────────────────────────────
        nxc = nx_cg_2d[out_idx]
        nyc = ny_cg_2d[out_idx]
        w_map, w_clean = defect_analysis(cfg["RING_RADIUS_2D"], nxc, nyc)
        winding_2d[out_idx]       = w_map.astype(np.float32)
        winding_clean_2d[out_idx] = w_clean.astype(np.float32)

    total_defects = int((np.abs(winding_clean_2d) > 0.1).sum())
    print(f"  Done in {time.time()-t_total:.1f}s  |  total 2D defect pixels: {total_defects}")

    return dict(
        nx_2d            = nx_2d,
        ny_2d            = ny_2d,
        nx_cg_2d         = nx_cg_2d,
        ny_cg_2d         = ny_cg_2d,
        S_2d             = S_2d,
        winding_2d       = winding_2d,
        winding_clean_2d = winding_clean_2d,
        slice_indices    = np.array(z_indices, dtype=np.int32),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(cfg: dict):
    print("=" * 72)
    print("  3D Nematic Analysis Pipeline")
    print("=" * 72)
    print(f"  Input  : {cfg['INPUT_TIFF']}")
    print(f"  Output : {cfg['OUTPUT_DIR']}")
    print(f"  Device : {cfg['DEVICE']}")
    if "cuda" in cfg["DEVICE"]:
        if th.cuda.is_available():
            idx = 0 if cfg["DEVICE"] == "cuda" else int(cfg["DEVICE"].split(":")[-1])
            gb  = th.cuda.get_device_properties(idx).total_memory / 1e9
            print(f"  GPU    : {th.cuda.get_device_name(idx)}  ({gb:.1f} GB)")
        else:
            print("  WARNING: CUDA requested but not available — falling back to CPU")
            cfg["DEVICE"] = "cpu"

    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    base = os.path.join(cfg["OUTPUT_DIR"], cfg["EXPERIMENT_NAME"])

    # ── load ─────────────────────────────────────────────────────────────────
    print("\n── Loading volume ────────────────────────────────────────────────────")
    vol = load_tiff(cfg["INPUT_TIFF"])

    meta = dict(
        mum_per_px_xy   = cfg["MUM_PER_PX_XY"],
        mum_per_px_z    = cfg["MUM_PER_PX_Z"],
        cgl_3d          = cfg["CGL_3D"],
        downsample_3d   = cfg["DOWNSAMPLE_3D"],
        q_scale_3d_um   = cfg["Q_SCALE_3D_UM"],
        ring_radius_3d  = cfg["RING_RADIUS_3D"],
        winding_thresh  = cfg["WINDING_THRESH"],
        min_line_length = cfg["MIN_LINE_LENGTH"],
        cgl_2d          = cfg["CGL_2D"],
        q_scale_2d_um   = cfg["Q_SCALE_2D_UM"],
        ring_radius_2d  = cfg["RING_RADIUS_2D"],
        slice_stride    = cfg["SLICE_STRIDE"],
        vol_shape_DHW   = np.array(vol.shape),
    )

    # ── 3D ───────────────────────────────────────────────────────────────────
    t_wall = time.time()
    results_3d = run_3d_pipeline(vol, cfg)

    print("\n── Saving 3D results ─────────────────────────────────────────────────")
    _save_pair(
        npz_path = base + "_3d.npz",
        mat_path = base + "_3d.mat",
        arrays   = results_3d,
        meta     = meta,
    )

    # ── 2D ───────────────────────────────────────────────────────────────────
    results_2d = run_2d_xy_pipeline(vol, cfg)

    print("\n── Saving 2D XY results ─────────────────────────────────────────────")
    _save_pair(
        npz_path = base + "_2d_xy.npz",
        mat_path = base + "_2d_xy.mat",
        arrays   = results_2d,
        meta     = meta,
    )

    print(f"\n✓ All done in {time.time()-t_wall:.1f}s")
    print(f"\n  Output files:")
    for ext in ["_3d.npz", "_3d.mat", "_2d_xy.npz", "_2d_xy.mat"]:
        path = base + ext
        size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
        print(f"    {path}  ({size_mb:.1f} MB)")

    # ── quick summary printout ────────────────────────────────────────────────
    r3 = results_3d
    r2 = results_2d
    print("\n  ── 3D summary ─────────────────────────────────────────────────────")
    print(f"    Volume shape          : {vol.shape}")
    print(f"    S_3d  mean / max      : {r3['S_3d'].mean():.3f} / {r3['S_3d'].max():.3f}")
    print(f"    Defect voxels (|w|>{cfg['WINDING_THRESH']:.2f}): "
          f"{int((np.abs(r3['charges3D']) > cfg['WINDING_THRESH']).sum())}")
    print("\n  ── 2D summary ─────────────────────────────────────────────────────")
    print(f"    Slices analysed       : {len(r2['slice_indices'])}")
    print(f"    S_2d  mean / max      : {r2['S_2d'].mean():.3f} / {r2['S_2d'].max():.3f}")
    n_plus  = int((r2['winding_clean_2d'] >  0.1).sum())
    n_minus = int((r2['winding_clean_2d'] < -0.1).sum())
    print(f"    +½ defect pixels      : {n_plus}")
    print(f"    -½ defect pixels      : {n_minus}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D nematic analysis pipeline")
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a JSON file whose keys override the built-in CFG dict."
    )
    args = parser.parse_args()

    cfg = dict(CFG)          # start from built-in defaults
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        cfg.update(overrides)
        print(f"  Config overrides loaded from: {args.config}")

    main(cfg)