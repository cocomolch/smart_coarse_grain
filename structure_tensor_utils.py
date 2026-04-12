import torch as th
import torch.nn.functional as F
import cv2 as cv
import scipy as sp
import numpy as np
from scipy.ndimage import gaussian_filter


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def binRadially(R, corr, num_bins=100):
    r_max = np.floor(np.max(R))
    increment = (r_max) / num_bins
    distances = np.arange(0, r_max, increment)
    binned_values = np.zeros(num_bins)
    counter = 0
    for r in distances:
        mask = np.logical_and(R >= r, R < r + increment)
        binned_values[counter] = np.mean(corr[mask])
        counter += 1
    return binned_values, distances


def autocorr2d_radial(native_theta, num_bins=200):
    """returns the radial average of the autocorrelation function"""
    corr = autocorr2d_fft(native_theta)
    [X, Y] = np.meshgrid(
        np.arange(0, corr.shape[1]), np.arange(0, corr.shape[0]), indexing="xy"
    )

    X_centered = X - corr.shape[0] / 2
    Y_centered = Y - corr.shape[1] / 2

    R, theta = cart2pol(X_centered, Y_centered)
    averageCorr, distances = binRadially(R, corr, num_bins)
    return averageCorr, distances


def autocorr2d_fft(img):
    """
    Fast 2D autocorrelation via the Wiener-Khinchin theorem:
    autocorr = IFFT2(|FFT2(img)|^2).
    Returns a map of size 2M*2N (full autocorr), centered at zero lag.
    """
    img = np.asarray(img, dtype=float)
    # subtract mean
    img -= img.mean()
    # pad to avoid circular wrap‐around if you want 'full' result
    M, N = img.shape
    fshape = (2 * M, 2 * N)
    # forward FFT
    F = np.fft.rfft2(img, s=fshape)
    # power spectrum
    P = np.abs(F) ** 2
    # inverse FFT to get autocorrelation
    corr = np.fft.irfft2(P, s=fshape)
    # shift zero‐lag to center
    corr = np.fft.fftshift(corr)

    # normalize
    center = (fshape[0] // 2, fshape[1] // 2)
    return corr / corr[center]


def getGaussianKernel(coarseGrainingLength):
    """
    Returns a Gaussian kernel for coarse graining.
    The kernel is of size (3 * coarseGrainingLength, 3 * coarseGrainingLength)
    """
    helper = np.zeros(
        (
            int(np.round(2.5 * coarseGrainingLength)),
            int(np.round(2.5 * coarseGrainingLength)),
        )
    )
    helper[int(np.floor(helper.shape[0] / 2)), int(np.floor(helper.shape[1] / 2))] = 1
    G = gaussian_filter(
        helper,
        sigma=coarseGrainingLength,
    )
    G = th.from_numpy(G)
    G = G.unsqueeze(0).unsqueeze(0)
    G = G.float()
    return G


def smart_structure_tensor(
    img_tensor,
    coarseGrainAverage="gaussian",
    coarseGrainingLength=4,
    downsample=2,
    method="inter_area",
):
    nativeDimensions = img_tensor.shape[2:]

    if coarseGrainAverage == "gaussian":
        # get gaussian kernel
        G = getGaussianKernel(coarseGrainingLength)
    elif coarseGrainAverage == "mean":
        # get avergae kernel
        G = np.ones((coarseGrainingLength, coarseGrainingLength)) / (
            coarseGrainingLength**2
        )
        G = th.from_numpy(G)
        G = G.unsqueeze(0).unsqueeze(0)
        G = G.float()

    G_small = getGaussianKernel(1)
    img_tensor = th.conv2d(img_tensor, G_small, padding="same")
    # get gradients
    sobel_x = (
        th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=th.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sobel_y = (
        th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=th.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    Ix = th.conv2d(img_tensor, sobel_x, padding="same")
    Iy = th.conv2d(img_tensor, sobel_y, padding="same")

    # construct structure tensor
    Ixx = th.conv2d(Ix * Ix, G, stride=downsample)
    Iyy = th.conv2d(Iy * Iy, G, stride=downsample)
    Ixy = th.conv2d(Ix * Iy, G, stride=downsample)

    tr = Ixx + Iyy
    diff = Ixx - Iyy
    enum = th.sqrt(diff**2 + 4 * Ixy**2)

    # smallest eigenvalue
    lambda2 = tr / 2 - th.sqrt(enum) / 2
    # corresponding eigenvector
    eigenvector2 = th.zeros(Ixx.shape[2], Ixx.shape[3], 2)
    eigenvector2[:, :, 0] = Ixy
    eigenvector2[:, :, 1] = lambda2 - Ixx
    # normalize eigenvectors
    norm = th.sqrt(eigenvector2[:, :, 0] ** 2 + eigenvector2[:, :, 1] ** 2)
    eigenvector2[:, :, 0] = eigenvector2[:, :, 0] / norm
    eigenvector2[:, :, 1] = eigenvector2[:, :, 1] / norm

    nx = eigenvector2[:, :, 0]
    ny = eigenvector2[:, :, 1]

    theta = th.acos(np.abs(nx))
    # theta = th.atan2(ny, nx)
    if method == "inter_area":
        theta_reconstruct = cv.resize(
            theta.numpy(),
            (nativeDimensions[1], nativeDimensions[0]),
            interpolation=cv.INTER_AREA,
        )
    elif method == "inter_nearest":
        theta_reconstruct = cv.resize(
            theta.numpy(),
            (nativeDimensions[1], nativeDimensions[0]),
            interpolation=cv.INTER_NEAREST_EXACT,
        )
    elif method == "inter_cubic":
        theta_reconstruct = cv.resize(
            theta.numpy(),
            (nativeDimensions[1], nativeDimensions[0]),
            interpolation=cv.INTER_CUBIC,
        )
    elif method == "inter_lanczos4":
        theta_reconstruct = cv.resize(
            theta.numpy(),
            (nativeDimensions[1], nativeDimensions[0]),
            interpolation=cv.INTER_LANCZOS4,
        )
    return theta_reconstruct




def smart_structure_tensor_gpu(
    img_tensor,
    coarseGrainAverage="gaussian",
    coarseGrainingLength=4,
    downsample=2,
    method="area",  # torch naming
    device="cuda",
):
    img_tensor = img_tensor.to(device).float()

    native_h, native_w = img_tensor.shape[2:]

    # -------------------------
    # Kernels
    # -------------------------
    def get_gaussian_kernel(size, sigma=None):
        if sigma is None:
            sigma = size / 3
        ax = th.arange(-size // 2 + 1., size // 2 + 1., device=device)
        xx, yy = th.meshgrid(ax, ax, indexing='ij')
        kernel = th.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    if coarseGrainAverage == "gaussian":
        G = get_gaussian_kernel(coarseGrainingLength)
    else:
        G = th.ones((1, 1, coarseGrainingLength, coarseGrainingLength), device=device)
        G /= coarseGrainingLength**2

    G_small = get_gaussian_kernel(3)

    # -------------------------
    # Pre-smoothing
    # -------------------------
    img_tensor = F.conv2d(img_tensor, G_small, padding="same")

    # -------------------------
    # Sobel gradients
    # -------------------------
    sobel_x = th.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=th.float32,
        device=device
    ).view(1, 1, 3, 3)

    sobel_y = th.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=th.float32,
        device=device
    ).view(1, 1, 3, 3)

    Ix = F.conv2d(img_tensor, sobel_x, padding=1)
    Iy = F.conv2d(img_tensor, sobel_y, padding=1)

    # -------------------------
    # Structure tensor
    # -------------------------
    Ixx = F.conv2d(Ix * Ix, G, stride=downsample)
    Iyy = F.conv2d(Iy * Iy, G, stride=downsample)
    Ixy = F.conv2d(Ix * Iy, G, stride=downsample)

    tr = Ixx + Iyy
    diff = Ixx - Iyy
    enum = th.sqrt(diff**2 + 4 * Ixy**2 + 1e-8)

    lambda2 = tr / 2 - enum / 2

    # -------------------------
    # Eigenvector (vectorized!)
    # -------------------------
    nx = Ixy
    ny = lambda2 - Ixx

    norm = th.sqrt(nx**2 + ny**2 + 1e-8)
    nx = nx / norm
    ny = ny / norm

    theta = th.acos(th.abs(nx).clamp(0, 1))

    # -------------------------
    # Resize on GPU
    # -------------------------
    theta = F.interpolate(
        theta,
        size=(native_h, native_w),
        mode={
            "area": "area",
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
        }[method],
        align_corners=False if method in ["bilinear", "bicubic"] else None,
    )

    return theta  



import torch as th
import torch.nn.functional as F


def get2DQTensor_gpu(theta, QtensorAverageScale, mum_per_px_2D):
    """
    theta: torch tensor [1,1,H,W] (radians)
    returns: nx_cg, ny_cg, S  (all on GPU)
    """

    device = theta.device

    # -------------------------
    # Director field
    # -------------------------
    nx = th.cos(theta)
    ny = th.sin(theta)

    # -------------------------
    # Q-tensor components
    # -------------------------
    Qxx = nx**2 - 0.5
    Qxy = nx * ny

    # -------------------------
    # Gaussian kernel
    # -------------------------
    sigma = QtensorAverageScale / mum_per_px_2D

    def get_gaussian_kernel(sigma, device):
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1

        ax = th.arange(-(size // 2), size // 2 + 1, device=device)
        xx, yy = th.meshgrid(ax, ax, indexing='ij')

        kernel = th.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        return kernel.view(1, 1, size, size)

    G = get_gaussian_kernel(sigma, device)

    # -------------------------
    # Coarse graining (GPU conv)
    # -------------------------
    Qxx_cg = F.conv2d(Qxx, G, padding="same")
    Qxy_cg = F.conv2d(Qxy, G, padding="same")

    # -------------------------
    # Order parameter
    # -------------------------
    S = 2 * th.sqrt(Qxx_cg**2 + Qxy_cg**2 + 1e-8)

    # -------------------------
    # Normalize Q-tensor
    # -------------------------
    Qxx_n = Qxx_cg / (S + 1e-8)
    Qxy_n = Qxy_cg / (S + 1e-8)

    # -------------------------
    # Back to director
    # -------------------------
    nx_cg = th.sqrt((Qxx_n + 0.5).clamp(min=0))
    ny_cg = th.sqrt((1 - nx_cg**2).clamp(min=0)) * th.sign(Qxy_n)

    return nx_cg, ny_cg, S


import torch as th
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _gaussian_kernel_3d_np(sigma, truncate=2.5):
    """Return a 3-D Gaussian kernel as a [1,1,D,H,W] torch tensor (float32)."""
    half = int(np.round(truncate * sigma))
    size = 2 * half + 1
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return th.from_numpy(kernel).float().reshape(1, 1, size, size, size)


def _gaussian_kernel_3d_gpu(sigma, device, truncate=2.5):
    """Same, but built directly on *device*."""
    half = int(np.round(truncate * sigma))
    ax = th.arange(-half, half + 1, device=device, dtype=th.float32)
    xx, yy, zz = th.meshgrid(ax, ax, ax, indexing="ij")
    kernel = th.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2])


# ═══════════════════════════════════════════════════════════════════
#  3-D Sobel kernels  (central-difference × triangle smoothing)
# ═══════════════════════════════════════════════════════════════════

def _sobel_kernels_3d(device=None):
    """
    Returns (Sx, Sy, Sz) each of shape [1,1,3,3,3].
    Convention: dim-0 = z, dim-1 = y, dim-2 = x
    """
    # 1-D pieces
    d = np.array([-1.0, 0.0, 1.0])   # derivative
    s = np.array([1.0, 2.0, 1.0])    # smoothing (triangle)

    # Sx: derivative along x (axis-2), smooth along y,z
    Sx = np.einsum("i,j,k->ijk", s, s, d)
    # Sy: derivative along y (axis-1), smooth along x,z
    Sy = np.einsum("i,j,k->ijk", s, d, s)
    # Sz: derivative along z (axis-0), smooth along x,y
    Sz = np.einsum("i,j,k->ijk", d, s, s)

    tensors = []
    for K in (Sx, Sy, Sz):
        t = th.from_numpy(K).float().reshape(1, 1, 3, 3, 3)
        if device is not None:
            t = t.to(device)
        tensors.append(t)
    return tensors


# ═══════════════════════════════════════════════════════════════════
#  Eigen-decomposition of 3×3 symmetric tensor (vectorized)
# ═══════════════════════════════════════════════════════════════════

def _eig_symmetric_3x3(Sxx, Syy, Szz, Sxy, Sxz, Syz):
    """
    Analytic eigenvalues + eigenvectors of a 3×3 symmetric matrix
    at every voxel.  Returns the eigenvector of the *smallest*
    eigenvalue (the fibre / ridge direction for a structure tensor).

    All inputs/outputs: same-shape tensors.
    Returns: (val_min, vx, vy, vz)
    """
    # Build [N, 3, 3] matrix
    orig_shape = Sxx.shape
    N = Sxx.numel()

    M = th.zeros(N, 3, 3, device=Sxx.device, dtype=Sxx.dtype)
    Sxx_f = Sxx.reshape(-1)
    Syy_f = Syy.reshape(-1)
    Szz_f = Szz.reshape(-1)
    Sxy_f = Sxy.reshape(-1)
    Sxz_f = Sxz.reshape(-1)
    Syz_f = Syz.reshape(-1)

    M[:, 0, 0] = Sxx_f
    M[:, 1, 1] = Syy_f
    M[:, 2, 2] = Szz_f
    M[:, 0, 1] = Sxy_f
    M[:, 1, 0] = Sxy_f
    M[:, 0, 2] = Sxz_f
    M[:, 2, 0] = Sxz_f
    M[:, 1, 2] = Syz_f
    M[:, 2, 1] = Syz_f

    # torch.linalg.eigh: eigenvalues in ascending order
    vals, vecs = th.linalg.eigh(M)  # vals [N,3], vecs [N,3,3]

    # smallest eigenvalue → column 0
    val_min = vals[:, 0].reshape(orig_shape)
    vx = vecs[:, 0, 0].reshape(orig_shape)
    vy = vecs[:, 1, 0].reshape(orig_shape)
    vz = vecs[:, 2, 0].reshape(orig_shape)

    return val_min, vx, vy, vz


# ═══════════════════════════════════════════════════════════════════
#  CPU version
# ═══════════════════════════════════════════════════════════════════

def smart_structure_tensor_3d(
    vol,
    coarse_grain_average="gaussian",
    coarse_graining_length=4,
    downsample=2,
    interpolation_method="trilinear",
):
    """
    Fast 3-D structure tensor with strided convolution + interpolation.

    Parameters
    ----------
    vol : np.ndarray (D, H, W)  or  torch.Tensor [1,1,D,H,W]
        Input volume (grayscale).
    coarse_grain_average : str
        'gaussian' or 'mean'.
    coarse_graining_length : int
        Size / sigma of the averaging kernel.
    downsample : int
        Stride used in the averaging convolutions. The key speedup.
    interpolation_method : str
        'trilinear', 'nearest'.

    Returns
    -------
    theta : np.ndarray (D, H, W)  — polar angle of the smallest-eigenvalue
            eigenvector (angle from z-axis).
    phi   : np.ndarray (D, H, W)  — azimuthal angle in x-y plane.
    vx, vy, vz : np.ndarray (D, H, W) — eigenvector components (upsampled).
    """

    # ---- prepare input ------------------------------------------------
    if isinstance(vol, np.ndarray):
        vol_t = th.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    else:
        vol_t = vol.float()
        if vol_t.ndim == 3:
            vol_t = vol_t.unsqueeze(0).unsqueeze(0)

    native_shape = vol_t.shape[2:]  # (D, H, W)

    # ---- kernels ------------------------------------------------------
    if coarse_grain_average == "gaussian":
        G = _gaussian_kernel_3d_np(coarse_graining_length)
    else:
        k = coarse_graining_length
        G = th.ones(1, 1, k, k, k) / k**3

    G_small = _gaussian_kernel_3d_np(1)

    # ---- pre-smooth ---------------------------------------------------
    vol_t = F.conv3d(vol_t, G_small, padding="same")

    # ---- gradients (Sobel 3-D) ----------------------------------------
    Sx, Sy, Sz = _sobel_kernels_3d()
    Ix = F.conv3d(vol_t, Sx, padding=1)
    Iy = F.conv3d(vol_t, Sy, padding=1)
    Iz = F.conv3d(vol_t, Sz, padding=1)

    # ---- structure tensor components (strided!) -----------------------
    pad = G.shape[-1] // 2
    Sxx = F.conv3d(Ix * Ix, G, stride=downsample, padding=pad)
    Syy = F.conv3d(Iy * Iy, G, stride=downsample, padding=pad)
    Szz = F.conv3d(Iz * Iz, G, stride=downsample, padding=pad)
    Sxy = F.conv3d(Ix * Iy, G, stride=downsample, padding=pad)
    Sxz = F.conv3d(Ix * Iz, G, stride=downsample, padding=pad)
    Syz = F.conv3d(Iy * Iz, G, stride=downsample, padding=pad)

    # ---- eigen-decomposition ------------------------------------------
    val_min, vx, vy, vz = _eig_symmetric_3x3(
        Sxx, Syy, Szz, Sxy, Sxz, Syz
    )

    # ---- angles at downsampled resolution ----------------------------
    # polar angle (from z-axis)
    theta_ds = th.acos(th.abs(vz).clamp(0, 1))
    # azimuthal angle
    phi_ds = th.atan2(vy, vx)

    # ---- upsample back to native resolution --------------------------
    mode = {
        "trilinear": "trilinear",
        "nearest": "nearest",
    }[interpolation_method]

    align = False if mode == "trilinear" else None

    theta_up = F.interpolate(
        theta_ds, size=native_shape, mode=mode, align_corners=align
    )
    phi_up = F.interpolate(
        phi_ds, size=native_shape, mode=mode, align_corners=align
    )

    # also upsample the eigenvector components for downstream Q-tensor use
    vx_up = F.interpolate(vx, size=native_shape, mode=mode, align_corners=align)
    vy_up = F.interpolate(vy, size=native_shape, mode=mode, align_corners=align)
    vz_up = F.interpolate(vz, size=native_shape, mode=mode, align_corners=align)

    # re-normalize after interpolation
    norm = th.sqrt(vx_up**2 + vy_up**2 + vz_up**2 + 1e-8)
    vx_up = vx_up / norm
    vy_up = vy_up / norm
    vz_up = vz_up / norm

    return (
        theta_up.squeeze().numpy(),
        phi_up.squeeze().numpy(),
        vx_up.squeeze().numpy(),
        vy_up.squeeze().numpy(),
        vz_up.squeeze().numpy(),
    )


# ═══════════════════════════════════════════════════════════════════
#  GPU version
# ═══════════════════════════════════════════════════════════════════

def smart_structure_tensor_3d_gpu(
    vol,
    coarse_grain_average="gaussian",
    coarse_graining_length=4,
    downsample=2,
    interpolation_method="trilinear",
    device="cuda",
):
    """
    GPU-accelerated 3-D structure tensor with strided convolution
    + interpolation.

    Parameters
    ----------
    vol : np.ndarray (D, H, W)  or  torch.Tensor
        Input volume.
    coarse_grain_average : str
        'gaussian' or 'mean'.
    coarse_graining_length : int
        Size / sigma of averaging kernel.
    downsample : int
        Stride for the averaging convolution.
    interpolation_method : str
        'trilinear' or 'nearest'.
    device : str
        'cuda', 'cuda:0', etc.

    Returns
    -------
    theta, phi, vx, vy, vz : torch.Tensor [1,1,D,H,W] on *device*
    """

    # ---- prepare input ------------------------------------------------
    if isinstance(vol, np.ndarray):
        vol_t = th.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    else:
        vol_t = vol.float()
        if vol_t.ndim == 3:
            vol_t = vol_t.unsqueeze(0).unsqueeze(0)

    vol_t = vol_t.to(device)
    native_shape = vol_t.shape[2:]

    # ---- kernels ------------------------------------------------------
    if coarse_grain_average == "gaussian":
        G = _gaussian_kernel_3d_gpu(coarse_graining_length, device)
    else:
        k = coarse_graining_length
        G = th.ones(1, 1, k, k, k, device=device) / k**3

    G_small = _gaussian_kernel_3d_gpu(1, device)

    # ---- pre-smooth ---------------------------------------------------
    vol_t = F.conv3d(vol_t, G_small, padding="same")

    # ---- gradients ----------------------------------------------------
    Sx, Sy, Sz = _sobel_kernels_3d(device=device)
    Ix = F.conv3d(vol_t, Sx, padding=1)
    Iy = F.conv3d(vol_t, Sy, padding=1)
    Iz = F.conv3d(vol_t, Sz, padding=1)

    # ---- structure tensor (strided!) ----------------------------------
    pad = G.shape[-1] // 2
    Sxx = F.conv3d(Ix * Ix, G, stride=downsample, padding=pad)
    Syy = F.conv3d(Iy * Iy, G, stride=downsample, padding=pad)
    Szz = F.conv3d(Iz * Iz, G, stride=downsample, padding=pad)
    Sxy = F.conv3d(Ix * Iy, G, stride=downsample, padding=pad)
    Sxz = F.conv3d(Ix * Iz, G, stride=downsample, padding=pad)
    Syz = F.conv3d(Iy * Iz, G, stride=downsample, padding=pad)

    # ---- free gradient memory early -----------------------------------
    del Ix, Iy, Iz
    th.cuda.empty_cache() if "cuda" in device else None

    # ---- eigen-decomposition ------------------------------------------
    val_min, vx, vy, vz = _eig_symmetric_3x3(
        Sxx, Syy, Szz, Sxy, Sxz, Syz
    )

    del Sxx, Syy, Szz, Sxy, Sxz, Syz
    th.cuda.empty_cache() if "cuda" in device else None

    # ---- angles -------------------------------------------------------
    theta_ds = th.acos(th.abs(vz).clamp(0, 1))
    phi_ds = th.atan2(vy, vx)

    # ---- upsample -----------------------------------------------------
    mode = interpolation_method
    align = False if mode == "trilinear" else None

    theta_up = F.interpolate(theta_ds, size=native_shape, mode=mode, align_corners=align)
    phi_up = F.interpolate(phi_ds, size=native_shape, mode=mode, align_corners=align)

    vx_up = F.interpolate(vx, size=native_shape, mode=mode, align_corners=align)
    vy_up = F.interpolate(vy, size=native_shape, mode=mode, align_corners=align)
    vz_up = F.interpolate(vz, size=native_shape, mode=mode, align_corners=align)

    # re-normalize
    norm = th.sqrt(vx_up**2 + vy_up**2 + vz_up**2 + 1e-8)
    vx_up /= norm
    vy_up /= norm
    vz_up /= norm

    return theta_up, phi_up, vx_up, vy_up, vz_up


# ═══════════════════════════════════════════════════════════════════
#  3-D Q-tensor (nematic order parameter) — GPU
# ═══════════════════════════════════════════════════════════════════

def get3DQTensor_gpu(vx, vy, vz, Q_average_scale, mum_per_px, device="cuda"):
    """
    Compute the 3-D nematic Q-tensor and scalar order parameter S
    from an eigenvector field.

    Parameters
    ----------
    vx, vy, vz : torch.Tensor [1,1,D,H,W]
        Director field components (on device).
    Q_average_scale : float
        Coarse-graining length in physical units (µm).
    mum_per_px : float
        Pixel size in µm.
    device : str

    Returns
    -------
    nx_cg, ny_cg, nz_cg : coarse-grained director components
    S : scalar nematic order parameter  (0 = isotropic, 1 = aligned)
    """
    vx = vx.to(device)
    vy = vy.to(device)
    vz = vz.to(device)

    # Q_ij = n_i n_j - (1/3) delta_ij   (traceless symmetric)
    Qxx = vx * vx - 1.0 / 3.0
    Qyy = vy * vy - 1.0 / 3.0
    Qzz = vz * vz - 1.0 / 3.0
    Qxy = vx * vy
    Qxz = vx * vz
    Qyz = vy * vz

    # Gaussian coarse-graining
    sigma = Q_average_scale / mum_per_px
    G = _gaussian_kernel_3d_gpu(sigma, device)

    Qxx_cg = F.conv3d(Qxx, G, padding="same")
    Qyy_cg = F.conv3d(Qyy, G, padding="same")
    Qzz_cg = F.conv3d(Qzz, G, padding="same")
    Qxy_cg = F.conv3d(Qxy, G, padding="same")
    Qxz_cg = F.conv3d(Qxz, G, padding="same")
    Qyz_cg = F.conv3d(Qyz, G, padding="same")

    # S = sqrt( (3/2) * Q_ij Q_ij )   (Frobenius-based scalar order)
    QQ = (
        Qxx_cg**2 + Qyy_cg**2 + Qzz_cg**2
        + 2 * Qxy_cg**2 + 2 * Qxz_cg**2 + 2 * Qyz_cg**2
    )
    S = th.sqrt(1.5 * QQ + 1e-8)

    # Recover director from largest eigenvector of <Q>
    _, nx_cg, ny_cg, nz_cg = _eig_symmetric_3x3(
        Qxx_cg, Qyy_cg, Qzz_cg, Qxy_cg, Qxz_cg, Qyz_cg
    )
    # For Q-tensor the *largest* eigenvalue gives the director,
    # but _eig_symmetric_3x3 returns the smallest. Use the largest:
    # eigh returns ascending order, so column 2 is the largest.
    # We need a small wrapper — or just re-call with negated tensor:
    _, nx_cg, ny_cg, nz_cg = _eig_symmetric_3x3(
        -Qxx_cg, -Qyy_cg, -Qzz_cg, -Qxy_cg, -Qxz_cg, -Qyz_cg
    )
    # negate back: eigenvectors are the same, just flip sign convention
    # (eigenvectors of -A for eigenvalue -λ are same as eigenvectors of A for λ)

    return nx_cg, ny_cg, nz_cg, S



import torch as th
import torch.nn.functional as F
import numpy as np
from skimage.morphology import disk, thin


def make_ring_kernel(radius):
    """Returns deltaX, deltaY, ring_kernel as numpy arrays."""
    large = disk(radius + 1).astype(int)
    small = disk(radius).astype(int)

    pad_y = large.shape[0] - small.shape[0]
    pad_x = large.shape[1] - small.shape[1]
    small_padded = np.pad(
        small,
        ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
        mode="constant",
    )

    ring_kernel = large - small_padded
    L = ring_kernel.shape[0]
    coords = np.arange(L) - L / 2
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    r = np.sqrt(X**2 + Y**2)

    X_masked = np.where(ring_kernel, X, 0.0)
    Y_masked = np.where(ring_kernel, Y, 0.0)

    deltaX = np.nan_to_num(Y_masked / r)
    deltaY = np.nan_to_num(-X_masked / r)

    return deltaX, deltaY, ring_kernel


def _to_conv_kernel(arr, device=None):
    """Numpy 2D array → [1,1,H,W] float32 torch tensor."""
    t = th.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def defect_analysis(radius, nx_cg, ny_cg, device=None):
    """
    Topological defect detection via winding-number convolution.

    Parameters
    ----------
    radius : int
        Ring kernel radius.
    nx_cg, ny_cg : np.ndarray (H, W)
        Coarse-grained director components.
    device : str or None
        None / 'cpu' → PyTorch on CPU.
        'cuda' / 'cuda:0' → PyTorch on GPU.

    Returns
    -------
    windingMap : np.ndarray (H, W)
    windingMap_clean : np.ndarray (H, W)  — ±0.5 at defect locations.
    """
    if device is None:
        device = "cpu"

    # ---- Q-tensor components ------------------------------------------
    Qxx = nx_cg**2 - 0.5
    Qxy = nx_cg * ny_cg

    # ---- gradients (numpy, cheap) ------------------------------------
    Qxx_y, Qxx_x = np.gradient(Qxx)
    Qxy_y, Qxy_x = np.gradient(Qxy)

    den = Qxx**2 + Qxy**2
    dtheta_dx = np.nan_to_num(0.5 * (Qxx * Qxy_x - Qxy * Qxx_x) / den)
    dtheta_dy = np.nan_to_num(0.5 * (Qxx * Qxy_y - Qxy * Qxx_y) / den)

    # ---- ring kernels → torch ----------------------------------------
    deltaX, deltaY, _ = make_ring_kernel(radius)
    kX = _to_conv_kernel(deltaX, device)
    kY = _to_conv_kernel(deltaY, device)
    pad = deltaX.shape[0] // 2

    # ---- winding map via torch conv2d --------------------------------
    dx_t = th.from_numpy(dtheta_dx.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    dy_t = th.from_numpy(dtheta_dy.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    windingMap_t = (
        F.conv2d(dy_t, kY, padding=pad) +
        F.conv2d(dx_t, kX, padding=pad)
    ) / (2 * np.pi)

    windingMap = windingMap_t.squeeze().cpu().numpy()

    # ---- defect localization -----------------------------------------
    plus_defects = thin(windingMap > 0.2)
    minus_defects = thin(windingMap < -0.2)

    windingMap_clean = np.zeros_like(windingMap)
    windingMap_clean[plus_defects] = +0.5
    windingMap_clean[minus_defects] = -0.5

    return windingMap, windingMap_clean



"""
3-D disclination detection.

Two approaches:
  1. `disclination_detection_3d`  — Q-tensor winding-number per orthogonal
     slice, accelerated with PyTorch conv2d  (matches your 2-D defect_analysis)
  2. `zapotocky_plaquette_3d`    — Zapotocky's plaquette sign-flip test
     (vectorised numpy, mirrors the C code)

Both return per-plane defect volumes and a combined charge volume.
"""

import numpy as np
import torch as th
import torch.nn.functional as F
from skimage.morphology import disk, thin, skeletonize_3d, binary_dilation


# ═══════════════════════════════════════════════════════════════════
#  Ring kernel (shared with 2-D code)
# ═══════════════════════════════════════════════════════════════════

def make_ring_kernel(radius):
    large = disk(radius + 1).astype(int)
    small = disk(radius).astype(int)
    pad_y = large.shape[0] - small.shape[0]
    pad_x = large.shape[1] - small.shape[1]
    small_padded = np.pad(
        small,
        ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
        mode="constant",
    )
    ring_kernel = large - small_padded
    L = ring_kernel.shape[0]
    coords = np.arange(L) - L / 2
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    r = np.sqrt(X**2 + Y**2)
    deltaX = np.nan_to_num(np.where(ring_kernel, Y, 0.0) / r)
    deltaY = np.nan_to_num(np.where(ring_kernel, -X, 0.0) / r)
    return deltaX, deltaY, ring_kernel


# ═══════════════════════════════════════════════════════════════════
#  2-D winding number (batched over slices)  — PyTorch accelerated
# ═══════════════════════════════════════════════════════════════════

def _winding_map_batch(nx_batch, ny_batch, radius, device):
    """
    Compute winding-number map for a batch of 2-D slices.

    nx_batch, ny_batch : np.ndarray  (N, H, W)
    Returns: winding_maps  np.ndarray  (N, H, W)
    """
    N, H, W = nx_batch.shape

    # Q-tensor
    Qxx = nx_batch**2 - 0.5
    Qxy = nx_batch * ny_batch

    # gradients per slice (axis -1 = x, axis -2 = y)
    Qxx_x = np.gradient(Qxx, axis=-1)
    Qxx_y = np.gradient(Qxx, axis=-2)
    Qxy_x = np.gradient(Qxy, axis=-1)
    Qxy_y = np.gradient(Qxy, axis=-2)

    den = Qxx**2 + Qxy**2
    dtheta_dx = np.nan_to_num(0.5 * (Qxx * Qxy_x - Qxy * Qxx_x) / den)
    dtheta_dy = np.nan_to_num(0.5 * (Qxx * Qxy_y - Qxy * Qxx_y) / den)

    # ring kernels → torch
    deltaX, deltaY, _ = make_ring_kernel(radius)
    kX = th.from_numpy(deltaX.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    kY = th.from_numpy(deltaY.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    pad = deltaX.shape[0] // 2

    # batch conv: treat N slices as batch dim, 1 channel
    dx_t = th.from_numpy(dtheta_dx.astype(np.float32)).unsqueeze(1).to(device)  # (N,1,H,W)
    dy_t = th.from_numpy(dtheta_dy.astype(np.float32)).unsqueeze(1).to(device)

    winding_t = (
        F.conv2d(dy_t, kY, padding=pad) + F.conv2d(dx_t, kX, padding=pad)
    ) / (2 * np.pi)

    return winding_t.squeeze(1).cpu().numpy()  # (N, H, W)


# ═══════════════════════════════════════════════════════════════════
#  Method 1: Winding-number per orthogonal slice (GPU-accelerated)
# ═══════════════════════════════════════════════════════════════════

def disclination_detection_3d(
    dx, dy, dz,
    radius=4,
    winding_thresh=0.2,
    min_line_length=10,
    device=None,
):
    """
    Detect 3-D disclination lines using Q-tensor winding numbers
    computed per orthogonal slice, accelerated with PyTorch conv2d.

    Parameters
    ----------
    dx, dy, dz : np.ndarray (D, H, W)
        Director-field components.
    radius : int
        Ring-kernel radius for winding-number convolution.
    winding_thresh : float
        Threshold on |winding| to identify defect pixels.
    min_line_length : int
        Connected components shorter than this are discarded.
    device : str or None
        'cpu', 'cuda', etc. None → 'cpu'.

    Returns
    -------
    charges3D : np.ndarray (D, H, W)
        Accumulated winding number from all 3 planes.
    xy_defects, xz_defects, yz_defects : np.ndarray (D, H, W)
        Per-plane raw winding maps.
    omega : np.ndarray (D, H, W, 3)
        Normal vector of the plane in which each defect was detected.
    """
    if device is None:
        device = "cpu"

    D, H, W = dx.shape
    charges3D = np.zeros((D, H, W), dtype=np.float32)
    xy_defects = np.zeros_like(charges3D)
    xz_defects = np.zeros_like(charges3D)
    yz_defects = np.zeros_like(charges3D)

    # --- XY slices (fix z) -------------------------------------------
    # in-plane components: (dx, dy), iterate over z
    nx_batch = dx.transpose(2, 0, 1)  # (W→N, D→H, H→W) — wrong, let me fix
    # For xy plane at each z-index: slice is dx[:,:,z], dy[:,:,z]
    nx_xy = np.stack([dx[:, :, z] for z in range(W)], axis=0)  # wait, that's wrong too

    # Let me be careful about axes:
    # Volume shape: (D, H, W) where dim-0=z, dim-1=y, dim-2=x
    # XY plane: fix z (dim-0). In-plane directors: dx (dim-2 component), dy (dim-1 component)
    nx_xy = np.ascontiguousarray(dx)          # (D, H, W) — each dx[z,:,:] is an xy slice
    ny_xy = np.ascontiguousarray(dy)

    # Normalize in-plane
    norm = np.sqrt(nx_xy**2 + ny_xy**2 + 1e-12)
    nx_xy_n = nx_xy / norm
    ny_xy_n = ny_xy / norm

    winding_xy = _winding_map_batch(nx_xy_n, ny_xy_n, radius, device)  # (D, H, W)
    xy_defects = winding_xy
    charges3D += winding_xy

    # --- XZ slices (fix y, dim-1) ------------------------------------
    # In-plane directors: dx (x-component), dz (z-component)
    nx_xz = np.ascontiguousarray(dx.transpose(1, 0, 2))  # (H, D, W) — each [y,:,:] is a (z,x) slice
    ny_xz = np.ascontiguousarray(dz.transpose(1, 0, 2))
    norm = np.sqrt(nx_xz**2 + ny_xz**2 + 1e-12)
    nx_xz_n = nx_xz / norm
    ny_xz_n = ny_xz / norm

    winding_xz_transposed = _winding_map_batch(nx_xz_n, ny_xz_n, radius, device)  # (H, D, W)
    winding_xz = winding_xz_transposed.transpose(1, 0, 2)  # back to (D, H, W)
    xz_defects = winding_xz
    charges3D += winding_xz

    # --- YZ slices (fix x, dim-2) ------------------------------------
    # In-plane directors: dy (y-component), dz (z-component)
    nx_yz = np.ascontiguousarray(dy.transpose(2, 0, 1))  # (W, D, H) — each [x,:,:] is a (z,y) slice
    ny_yz = np.ascontiguousarray(dz.transpose(2, 0, 1))
    norm = np.sqrt(nx_yz**2 + ny_yz**2 + 1e-12)
    nx_yz_n = nx_yz / norm
    ny_yz_n = ny_yz / norm

    winding_yz_transposed = _winding_map_batch(nx_yz_n, ny_yz_n, radius, device)  # (W, D, H)
    winding_yz = winding_yz_transposed.transpose(1, 2, 0)  # back to (D, H, W)
    yz_defects = winding_yz
    charges3D += winding_yz

    # --- Omega volume (plane normal for each defect) ------------------
    omega = np.zeros((D, H, W, 3), dtype=np.float32)
    # z-normal for xy defects, y-normal for xz, x-normal for yz
    omega[np.abs(xy_defects) > winding_thresh, 2] = 1.0  # [0,0,1]
    omega[np.abs(xz_defects) > winding_thresh, 1] = 1.0  # [0,1,0]
    omega[np.abs(yz_defects) > winding_thresh, 0] = 1.0  # [1,0,0]

    return charges3D, xy_defects, xz_defects, yz_defects, omega


# ═══════════════════════════════════════════════════════════════════
#  Method 2: Zapotocky plaquette sign-flip test (vectorised numpy)
# ═══════════════════════════════════════════════════════════════════

def _sequential_align_and_check(n0, n1, n2, n3):
    """
    Zapotocky algorithm for a batch of plaquettes.

    n0..n3: arrays of shape (..., 3) — directors at 4 corners
            ordered counter-clockwise around the plaquette.

    Returns: bool array (...) — True where a disclination pierces.

    Algorithm: sequentially flip n_{i+1} if it's closer to -n_i than n_i,
    then check if closing the loop (n3 → n0) requires a flip.
    """
    def flip_if_closer(a, b):
        """Flip b where dot(a,b) < 0."""
        dot = np.sum(a * b, axis=-1, keepdims=True)
        return np.where(dot < 0, -b, b)

    n1 = flip_if_closer(n0, n1)
    n2 = flip_if_closer(n1, n2)
    n3 = flip_if_closer(n2, n3)

    # Check closure: is n0 closer to -n3 than n3?
    dot_close = np.sum(n3 * n0, axis=-1)
    return dot_close < 0  # True → disclination pierces this plaquette


def zapotocky_plaquette_3d(dx, dy, dz):
    """
    Zapotocky's plaquette method for 3-D disclination detection.
    Vectorised numpy implementation (no loops over voxels).

    Parameters
    ----------
    dx, dy, dz : np.ndarray (D, H, W)
        Director field components.

    Returns
    -------
    charges3D : np.ndarray (D, H, W)
        Sum of plaquette piercings from all 3 planes.
    xy_pierce, yz_pierce, zx_pierce : np.ndarray (D, H, W) bool
        Per-plane plaquette piercings.
    """
    D, H, W = dx.shape

    # Stack director into (D, H, W, 3)
    n = np.stack([dx, dy, dz], axis=-1)

    # Normalize
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / (norm + 1e-12)

    xy_pierce = np.zeros((D, H, W), dtype=bool)
    yz_pierce = np.zeros((D, H, W), dtype=bool)
    zx_pierce = np.zeros((D, H, W), dtype=bool)

    # --- XY plaquettes: corners at (z, y, x), (z, y, x+1), (z, y+1, x+1), (z, y+1, x)
    n0 = n[:, :-1, :-1, :]    # (D, H-1, W-1, 3)
    n1 = n[:, :-1, 1:,  :]
    n2 = n[:, 1:,  1:,  :]
    n3 = n[:, 1:,  :-1, :]
    result = _sequential_align_and_check(n0, n1, n2, n3)
    xy_pierce[:, :-1, :-1] = result

    # --- YZ plaquettes: corners at (z, y, x), (z, y+1, x), (z+1, y+1, x), (z+1, y, x)
    n0 = n[:-1, :-1, :, :]    # (D-1, H-1, W, 3)
    n1 = n[:-1, 1:,  :, :]
    n2 = n[1:,  1:,  :, :]
    n3 = n[1:,  :-1, :, :]
    result = _sequential_align_and_check(n0, n1, n2, n3)
    yz_pierce[:-1, :-1, :] = result

    # --- ZX plaquettes: corners at (z, y, x), (z+1, y, x), (z+1, y, x+1), (z, y, x+1)
    n0 = n[:-1, :, :-1, :]    # (D-1, H, W-1, 3)
    n1 = n[1:,  :, :-1, :]
    n2 = n[1:,  :, 1:,  :]
    n3 = n[:-1, :, 1:,  :]
    result = _sequential_align_and_check(n0, n1, n2, n3)
    zx_pierce[:-1, :, :-1] = result

    charges3D = xy_pierce.astype(np.float32) + yz_pierce.astype(np.float32) + zx_pierce.astype(np.float32)

    return charges3D, xy_pierce, yz_pierce, zx_pierce


# ═══════════════════════════════════════════════════════════════════
#  Omega vector + beta angle (post-processing)
# ═══════════════════════════════════════════════════════════════════

def compute_omega_and_beta(
    charges3D,
    xy_defects, xz_defects, yz_defects,
    defect_thresh=0.2,
    min_line_length=10,
):
    """
    Compute the omega (plane normal) and beta (angle between
    disclination tangent and omega) for detected disclination lines.

    Parameters
    ----------
    charges3D : np.ndarray (D, H, W)
    xy_defects, xz_defects, yz_defects : np.ndarray (D, H, W)
        Per-plane defect maps (from either method).
    defect_thresh : float
        Threshold to binarise defect maps.
    min_line_length : int
        Minimum connected-component size to keep.

    Returns
    -------
    omega_volume : np.ndarray (D, H, W, 3)
    beta_volume  : np.ndarray (D, H, W)
    skel         : np.ndarray (D, H, W) bool — skeletonized defect lines
    """
    from scipy.ndimage import label, binary_dilation as scipy_dilate

    D, H, W = charges3D.shape

    # Omega: assign plane normal based on which plane detected the defect
    omega_volume = np.zeros((D, H, W, 3), dtype=np.float32)

    xy_mask = np.abs(xy_defects) > defect_thresh
    xz_mask = np.abs(xz_defects) > defect_thresh
    yz_mask = np.abs(yz_defects) > defect_thresh

    # Skeletonize each plane's detections
    struct = np.ones((3, 3, 3), dtype=bool)

    xy_skel = skeletonize_3d(scipy_dilate(xy_mask, struct).astype(np.uint8))
    xz_skel = skeletonize_3d(scipy_dilate(xz_mask, struct).astype(np.uint8))
    yz_skel = skeletonize_3d(scipy_dilate(yz_mask, struct).astype(np.uint8))

    omega_volume[xy_skel > 0, 2] = 1.0   # z-normal
    omega_volume[xz_skel > 0, 1] = 1.0   # y-normal
    omega_volume[yz_skel > 0, 0] = 1.0   # x-normal

    # Combined skeleton
    defect_mask = np.abs(charges3D) > defect_thresh
    defect_dilated = scipy_dilate(defect_mask, struct)
    skel = skeletonize_3d(defect_dilated.astype(np.uint8)) > 0

    # Remove small components
    labeled, n_cc = label(skel)
    for cc_id in range(1, n_cc + 1):
        if np.sum(labeled == cc_id) < min_line_length:
            skel[labeled == cc_id] = False

    # Beta: angle between local tangent and omega
    beta_volume = np.zeros((D, H, W), dtype=np.float32)
    labeled, n_cc = label(skel)

    for cc_id in range(1, n_cc + 1):
        coords = np.argwhere(labeled == cc_id)  # (N, 3)
        if len(coords) < 3:
            continue

        for idx in range(len(coords)):
            p = coords[idx]
            # find 2 nearest neighbours for tangent estimation
            dists = np.sum((coords - p) ** 2, axis=1)
            dists[idx] = np.inf
            nn_indices = np.argsort(dists)[:2]
            tangent = coords[nn_indices[0]].astype(float) - coords[nn_indices[1]].astype(float)
            t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-8:
                continue
            tangent /= t_norm

            omega_vec = omega_volume[p[0], p[1], p[2]]
            o_norm = np.linalg.norm(omega_vec)
            if o_norm < 1e-8:
                continue
            omega_vec = omega_vec / o_norm

            dot = np.clip(np.abs(np.dot(tangent, omega_vec)), 0, 1)
            beta_volume[p[0], p[1], p[2]] = np.arccos(dot)

    return omega_volume, beta_volume, skel