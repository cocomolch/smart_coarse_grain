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
sys.path.insert(0, os.path.dirname('C:/Users/pgotthe/Documents/WSINematics/smart_coarse_grain/smart_coarse_grain/'))
from structure_tensor_utils import (
    # 3D
    smart_structure_tensor_3d_gpu,
    get3DQTensor_gpu,
    disclination_detection_3d,
    compute_omega_and_beta,
    # helpers exposed by the module
    _gaussian_kernel_3d_gpu,
    _eig_symmetric_3x3,
    # 2D
    smart_structure_tensor_gpu,
    get2DQTensor_gpu,
    defect_analysis,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← edit this block; everything else runs automatically
# ═══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    # ── I/O ──────────────────────────────────────────────────────────────────
    INPUT_TIFF        = "/path/to/your/volume.tif",   # multi-page TIFF (Z, Y, X)
    OUTPUT_DIR        = "/path/to/output",             # created automatically
    EXPERIMENT_NAME   = "my_sample",                   # prefix for saved files

    # ── Physical calibration ─────────────────────────────────────────────────
    MUM_PER_PX_XY     = 0.5,    # lateral pixel size  [µm/px]
    MUM_PER_PX_Z      = 1.0,    # axial   pixel size  [µm/slice]

    # ── 3D structure tensor ───────────────────────────────────────────────────
    CGL_3D            = 4,      # coarse-graining kernel size [voxels]
    DOWNSAMPLE_3D     = 2,      # strided-conv factor (speed vs. resolution)
    CG_METHOD_3D      = "gaussian",  # "gaussian" | "mean"

    # ── 3D Q-tensor coarse-graining ───────────────────────────────────────────
    Q_SCALE_3D_UM     = 5.0,    # Gaussian σ for Q-tensor averaging [µm]

    # ── 3D disclination detection ─────────────────────────────────────────────
    RING_RADIUS_3D    = 4,      # ring-kernel radius for winding-number [voxels]
    WINDING_THRESH    = 0.2,    # |winding| threshold to flag a defect voxel
    MIN_LINE_LENGTH   = 10,     # discard skeleton components shorter than this

    # ── 2D XY-slice analysis ──────────────────────────────────────────────────
    CGL_2D            = 4,      # coarse-graining kernel size [pixels]
    DOWNSAMPLE_2D     = 2,
    Q_SCALE_2D_UM     = 5.0,    # Gaussian σ for 2D Q-tensor [µm]
    RING_RADIUS_2D    = 4,      # winding-number ring kernel [pixels]
    SLICE_STRIDE      = 1,      # 1 = every slice, N = every Nth slice

    # ── Hardware ──────────────────────────────────────────────────────────────
    DEVICE            = "cuda",  # "cuda" | "cuda:0" | "cpu"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

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


def _save_pair(npz_path: str, mat_path: str, arrays: dict, meta: dict):
    """Save dict of arrays to .npz and .mat; warns if any value is too large for mat."""
    # .npz (lossless, no size limit)
    np.savez_compressed(npz_path, **arrays, **{f"meta_{k}": np.array(v) for k, v in meta.items()})
    print(f"  Saved: {npz_path}")

    # .mat — scipy savemat uses HDF5 (v7.3) for large files if format='5' exceeds 2 GB;
    # here we use the default v5 and fall back to individual key warnings.
    mat_dict = {}
    for k, v in arrays.items():
        arr = np.array(v)
        # MATLAB variable names: no leading digit, no hyphens
        safe_k = k.lstrip("0123456789").replace("-", "_")
        mat_dict[safe_k] = arr
    for k, v in meta.items():
        mat_dict[f"meta_{k}"] = np.array(v)
    try:
        sio.savemat(mat_path, mat_dict, do_compression=True)
        print(f"  Saved: {mat_path}")
    except Exception as exc:
        warnings.warn(f"  .mat save failed ({exc}); try splitting output or using HDF5.")


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

    nx_cg_t, ny_cg_t, nz_cg_t, S_t = get3DQTensor_gpu(
        vx_t, vy_t, vz_t,
        Q_average_scale = cfg["Q_SCALE_3D_UM"],
        mum_per_px      = cfg["MUM_PER_PX_XY"],
        device          = device,
    )

    nx_cg = nx_cg_t.squeeze().cpu().numpy()
    ny_cg = ny_cg_t.squeeze().cpu().numpy()
    nz_cg = nz_cg_t.squeeze().cpu().numpy()
    S_3d  = S_t.squeeze().cpu().numpy()
    print(f"  Done in {time.time()-t0:.1f}s  |  S mean={S_3d.mean():.3f}  max={S_3d.max():.3f}")

    # ── Recover & store full Q-tensor components after CG ─────────────────────
    # We replicate the averaging on the raw Q, giving us the properly averaged
    # tensor components (not just recomputed from CG directors).
    print("\n── Computing CG Q-tensor components ─────────────────────────────────")
    t0 = time.time()
    sigma = cfg["Q_SCALE_3D_UM"] / cfg["MUM_PER_PX_XY"]
    G_q = _gaussian_kernel_3d_gpu(sigma, device)

    def _cg(arr_np):
        t = th.from_numpy(arr_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        return F.conv3d(t, G_q, padding="same").squeeze().cpu().numpy()

    # raw Q components from raw directors
    Qxx_cg = _cg(vx * vx - 1.0/3.0)
    Qyy_cg = _cg(vy * vy - 1.0/3.0)
    Qzz_cg = _cg(vz * vz - 1.0/3.0)
    Qxy_cg = _cg(vx * vy)
    Qxz_cg = _cg(vx * vz)
    Qyz_cg = _cg(vy * vz)
    print(f"  Done in {time.time()-t0:.1f}s")

    # free GPU memory
    del vx_t, vy_t, vz_t, nx_cg_t, ny_cg_t, nz_cg_t, S_t, G_q
    if "cuda" in device:
        th.cuda.empty_cache()

    # ── 3D disclination detection ─────────────────────────────────────────────
    print("\n── 3D disclination detection (winding-number, GPU) ──────────────────")
    t0 = time.time()

    charges3D, winding_xy, winding_xz, winding_yz, omega_raw = disclination_detection_3d(
        nx_cg, ny_cg, nz_cg,
        radius          = cfg["RING_RADIUS_3D"],
        winding_thresh  = cfg["WINDING_THRESH"],
        min_line_length = cfg["MIN_LINE_LENGTH"],
        device          = device,
    )

    n_defect_vox = int((np.abs(charges3D) > cfg["WINDING_THRESH"]).sum())
    print(f"  Done in {time.time()-t0:.1f}s  |  defect voxels: {n_defect_vox}")

    # ── Omega / beta / skeleton post-processing ───────────────────────────────
    print("\n── Omega vector + beta angle + skeleton ─────────────────────────────")
    t0 = time.time()

    omega_3d, beta_3d, skel_3d = compute_omega_and_beta(
        charges3D,
        winding_xy, winding_xz, winding_yz,
        defect_thresh   = cfg["WINDING_THRESH"],
        min_line_length = cfg["MIN_LINE_LENGTH"],
    )
    n_skel = int(skel_3d.sum())
    print(f"  Done in {time.time()-t0:.1f}s  |  skeleton voxels: {n_skel}")

    return dict(
        # raw structure tensor
        vx=vx.astype(np.float32),
        vy=vy.astype(np.float32),
        vz=vz.astype(np.float32),
        theta_3d=theta_3d.astype(np.float32),
        phi_3d=phi_3d.astype(np.float32),
        # Q-tensor CG
        nx_cg=nx_cg.astype(np.float32),
        ny_cg=ny_cg.astype(np.float32),
        nz_cg=nz_cg.astype(np.float32),
        S_3d=S_3d.astype(np.float32),
        # disclination
        charges3D=charges3D.astype(np.float32),
        winding_xy=winding_xy.astype(np.float32),
        winding_xz=winding_xz.astype(np.float32),
        winding_yz=winding_yz.astype(np.float32),
        omega_3d=omega_3d.astype(np.float32),
        beta_3d=beta_3d.astype(np.float32),
        skel_3d=skel_3d.astype(np.uint8),  # bool → uint8 for MATLAB compat
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
    print(f"    Skeleton voxels       : {int(r3['skel_3d'].sum())}")
    print(f"    Beta angle mean (skel): "
          f"{r3['beta_3d'][r3['skel_3d'] > 0].mean():.3f} rad"
          if r3['skel_3d'].sum() > 0 else "    Beta angle mean (skel): n/a")
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