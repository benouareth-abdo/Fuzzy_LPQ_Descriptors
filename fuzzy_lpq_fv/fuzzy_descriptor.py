"""
Section 5 — Fuzzy soft-histogram extraction (Algorithms 4-5).

Rather than enumerating all 2^n candidate codes explicitly, the per-pixel
vote weight vector is built by successive Kronecker (outer) products of
[1-mu_k, mu_k] across bits (Algorithm 5), vectorized here over all pixels
of a grid cell at once. We also expose the raw per-pixel fuzzy descriptor
(`local` output) for later Fisher Vector encoding.
"""

import numpy as np

from .crisp import _grid_slices
from .stft import stft_coefficients


def fuzzy_soft_histogram_binary(memberships, grid=(4, 4)):
    """memberships: (n, H, W) in [0,1] for the binary (K=2) case."""
    n, H, W = memberships.shape
    hists = []
    for sy, sx in _grid_slices(H, W, grid):
        flat = memberships[:, sy, sx].reshape(n, -1)
        w = np.stack([1 - flat[0], flat[0]], axis=0)
        for k in range(1, n):
            mk = flat[k]
            w = np.concatenate([w * (1 - mk)[None, :], w * mk[None, :]], axis=0)
        h = w.sum(axis=1)
        h = h / (h.sum() + 1e-12)
        hists.append(h)
    return np.concatenate(hists)


def fuzzy_local_descriptor_binary(memberships):
    """Per-pixel raw fuzzy descriptor vector (Fisher Vector input): shape (H*W, n)."""
    n, H, W = memberships.shape
    return memberships.reshape(n, -1).T


def extract_fuzzy_descriptor(image, configs, quantizers, grid=(4, 4)):
    hist_parts, local_parts = [], []
    for cfg, quant in zip(configs, quantizers):
        coeffs = stft_coefficients(image, **cfg)
        mu = quant.membership(coeffs)
        hist_parts.append(fuzzy_soft_histogram_binary(mu, grid=grid))
        local_parts.append(fuzzy_local_descriptor_binary(mu))
    return np.concatenate(hist_parts), np.concatenate(local_parts, axis=1)
