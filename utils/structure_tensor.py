import torch as th
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter

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
    #ny = eigenvector2[:, :, 1]

    theta = th.acos(np.abs(nx))
    if downsample != 1:
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
    else:
        theta_reconstruct = theta

    return theta_reconstruct