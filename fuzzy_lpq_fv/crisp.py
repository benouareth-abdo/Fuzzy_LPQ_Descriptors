"""
Section 2 — Crisp LPQ / LPQ6 / MP-LPQ (Algorithm 2).

* LPQ: 4 frequency points, M=17 -> 8 coefficients -> 256-bin histogram.
* LPQ6: 6 frequency points (added second ring), larger window M=19 -> 12
  coefficients.
* MP-LPQ: P=3 independent frequency-point configurations, histograms
  concatenated.

Histograms are computed on a spatial grid of cells (default 4x4) and
concatenated, as is standard practice for locally-aggregated texture
descriptors.
"""

import numpy as np

from .stft import stft_coefficients

LPQ_CONFIG = dict(M=17, freq_points=((1, 0), (0, 1), (1, 1), (1, -1)))
LPQPLUS_CONFIG = dict(M=19, freq_points=((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)))
LPQPLUSPLUS_CONFIGS = [
    dict(M=15, freq_points=((1, 0), (0, 1), (1, 1), (1, -1))),
    dict(M=17, freq_points=((1, 0), (0, 1), (1, 1), (1, -1))),
    dict(M=19, freq_points=((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2))),
]

CONFIG_MAP = {"LPQ": [LPQ_CONFIG], "LPQ6": [LPQ6_CONFIG], "MP-LPQ": MP_LPQ_CONFIGS}


def crisp_code_map(coeffs):
    bits = (coeffs >= 0).astype(np.uint32)
    n = bits.shape[0]
    weights = (2 ** np.arange(n)).reshape(n, 1, 1)
    return (bits * weights).sum(axis=0)


def _grid_slices(H, W, grid):
    gy, gx = grid
    ys = np.linspace(0, H, gy + 1).astype(int)
    xs = np.linspace(0, W, gx + 1).astype(int)
    for a_ in range(gy):
        for b_ in range(gx):
            yield slice(ys[a_], ys[a_ + 1]), slice(xs[b_], xs[b_ + 1])


def crisp_histogram(code_map, n_bits, grid=(4, 4)):
    H, W = code_map.shape
    n_codes = 2 ** n_bits
    hists = []
    for sy, sx in _grid_slices(H, W, grid):
        cell = code_map[sy, sx].ravel()
        h = np.bincount(cell, minlength=n_codes).astype(np.float64)
        h = h / (h.sum() + 1e-12)
        hists.append(h)
    return np.concatenate(hists)


def extract_crisp_descriptor(image, configs, grid=(4, 4)):
    parts = []
    for cfg in configs:
        coeffs = stft_coefficients(image, **cfg)
        code = crisp_code_map(coeffs)
        parts.append(crisp_histogram(code, n_bits=coeffs.shape[0], grid=grid))
    return np.concatenate(parts)
