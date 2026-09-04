"""
Section 1 — STFT phase coefficient extraction (Algorithm 1).

For each 2-D frequency point u_i = a * (i, j) with a = 1/M, we compute the
local short-term Fourier transform via a 2-D convolution with a complex
exponential kernel (the separable 1-D formulation from the paper is
mathematically equivalent; the direct 2-D form below is used here for
clarity). Real and imaginary parts of each frequency point give two scalar
coefficient maps.
"""

import numpy as np
from scipy.signal import fftconvolve


def stft_coefficients(image, M=17, freq_points=((1, 0), (0, 1), (1, 1), (1, -1))):
    """Return an array of shape (n=2*len(freq_points), H, W) of STFT phase coefficients."""
    image = image.astype(np.float64)
    a = 1.0 / M
    r = M // 2
    ys = np.arange(-r, r + 1)
    Y1, Y2 = np.meshgrid(ys, ys, indexing="ij")
    coeffs = []
    for (i, j) in freq_points:
        u1, u2 = a * i, a * j
        kernel = np.exp(-1j * 2 * np.pi * (u1 * Y1 + u2 * Y2))
        F = fftconvolve(image, kernel, mode="same")
        coeffs.append(F.real)
        coeffs.append(F.imag)
    return np.stack(coeffs, axis=0)
