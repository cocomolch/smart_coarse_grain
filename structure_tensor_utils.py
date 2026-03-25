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

    return theta  # stays on GPU!