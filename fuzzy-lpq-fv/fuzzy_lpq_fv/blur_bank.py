"""
Section 6 — Blur degradation bank (Gaussian / motion / disk, matching
MATLAB's `fspecial`).

Used to build the blur-degraded test partitions of the evaluation
protocol.
"""

import numpy as np
from scipy.signal import fftconvolve


def fspecial_gaussian(hsize=7, sigma=1.0):
    r = hsize // 2
    ys = np.arange(-r, r + 1)
    Y1, Y2 = np.meshgrid(ys, ys, indexing="ij")
    h = np.exp(-(Y1 ** 2 + Y2 ** 2) / (2 * sigma ** 2))
    return h / h.sum()


def fspecial_disk(radius=3):
    r = int(np.ceil(radius))
    ys = np.arange(-r, r + 1)
    Y1, Y2 = np.meshgrid(ys, ys, indexing="ij")
    mask = (Y1 ** 2 + Y2 ** 2) <= radius ** 2
    h = mask.astype(np.float64)
    return h / h.sum()


def fspecial_motion(length=9, angle=0.0):
    length = max(int(round(length)), 1)
    theta = np.deg2rad(angle)
    half = (length - 1) / 2.0
    size = int(2 * np.ceil(half) + 1)
    h = np.zeros((size, size))
    center = size // 2
    for t in np.linspace(-half, half, max(length * 4, 2)):
        x = int(round(center + t * np.cos(theta)))
        y = int(round(center - t * np.sin(theta)))
        if 0 <= y < size and 0 <= x < size:
            h[y, x] = 1.0
    if h.sum() == 0:
        h[center, center] = 1.0
    return h / h.sum()


def apply_blur(image, kernel):
    return fftconvolve(image.astype(np.float64), kernel, mode="same")


BLUR_BANK = {
    "gaussian_3x3_sigma1": lambda: fspecial_gaussian(hsize=3, sigma=1),
    "gaussian_3x3_sigma2": lambda: fspecial_gaussian(hsize=3, sigma=2),
    "gaussian_3x3_sigma3": lambda: fspecial_gaussian(hsize=3, sigma=3),
    "motion_len8_angle0": lambda: fspecial_motion(length=8.0, angle=0),
    "motion_len9_angle0": lambda: fspecial_motion(length=9.0, angle=0),
    "disk_radius2": lambda: fspecial_disk(radius=2),
    "disk_radius3": lambda: fspecial_disk(radius=3),
}