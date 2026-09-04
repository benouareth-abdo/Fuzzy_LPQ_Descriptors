"""
Section 3 — Fuzzy membership functions (Eqs. 3-4) and
Section 4 — Adaptive fuzzy partition learning (Algorithm 3).

* SigmoidFuzzyQuantizer: binary (K=2) case, Eq. (3). As tau_k -> 0 it
  recovers the crisp sign quantization exactly.
* GaussianRBFPartition: general K-level case, Eq. (4), a normalized
  Gaussian RBF partition of unity (numerically stabilized with a
  log-sum-exp shift).
* AdaptiveFuzzyPartitionLearner: an EM-style gradient-on-center
  procedure. The E-step computes soft assignments of pooled training
  coefficients to the current partition; the M-step re-estimates each
  center as the membership-weighted mean, and gradient-ascends the
  bandwidth/temperature toward a target partition entropy (a proxy for
  the entropy-vs-margin trade-off of the paper), clipped to avoid
  degeneracy or collapse to a uniform partition.
"""

import numpy as np


class SigmoidFuzzyQuantizer:
    """Binary (K=2) fuzzy quantizer, Eq. (3). c=0, tau->0 recovers crisp LPQ."""

    def __init__(self, n_coeffs):
        self.c = np.zeros(n_coeffs)
        self.tau = np.ones(n_coeffs)

    def membership(self, coeffs):
        c = self.c.reshape(-1, 1, 1)
        tau = self.tau.reshape(-1, 1, 1)
        return 1.0 / (1.0 + np.exp(-(coeffs - c) / np.maximum(tau, 1e-6)))


class GaussianRBFPartition:
    """K-level fuzzy partition, Eq. (4)."""

    def __init__(self, n_coeffs, K):
        self.K = K
        self.centers = np.zeros((n_coeffs, K))
        self.sigmas = np.ones((n_coeffs, K))

    def membership(self, coeffs):
        n, H, W = coeffs.shape
        mus = np.zeros((n, self.K, H, W))
        for k in range(n):
            c = self.centers[k].reshape(self.K, 1, 1)
            s = self.sigmas[k].reshape(self.K, 1, 1)
            neg_d2 = -((coeffs[k][None] - c) ** 2) / (2 * s ** 2 + 1e-12)
            neg_d2 = neg_d2 - neg_d2.max(axis=0, keepdims=True)  # log-sum-exp stability
            unnorm = np.exp(neg_d2)
            mus[k] = unnorm / (unnorm.sum(axis=0, keepdims=True) + 1e-12)
        return mus


class AdaptiveFuzzyPartitionLearner:
    def __init__(self, K=2, lr=60.0, max_iter=30, sigma_min=1e-3, tol=1e-4):
        self.K = K
        self.lr = lr
        self.max_iter = max_iter
        self.sigma_min = sigma_min
        self.tol = tol

    def fit_one_coefficient(self, samples):
        samples = samples.ravel()
        if self.K == 2:
            scale = np.std(samples) + 1e-6
            c = np.median(samples)
            tau = 0.5 * scale
            target_entropy = 0.85  # normalized binary entropy target in (0,1]
            tau_min, tau_max = 1e-3 * scale, 3.0 * scale
            for _ in range(self.max_iter):
                mu = 1.0 / (1.0 + np.exp(-(samples - c) / max(tau, 1e-6)))
                new_c = np.average(samples, weights=mu * (1 - mu) + 1e-6)
                entropy = np.mean(
                    -(mu * np.log(mu + 1e-12) + (1 - mu) * np.log(1 - mu + 1e-12))
                ) / np.log(2)
                grad = target_entropy - entropy
                new_tau = np.clip(tau * (1.0 + self.lr * grad), tau_min, tau_max)
                shift = abs(new_c - c) + abs(new_tau - tau)
                c, tau = new_c, new_tau
                if shift < self.tol:
                    break
            return {"c": c, "tau": tau}
        else:
            quantiles = np.linspace(0, 1, self.K + 2)[1:-1]
            centers = np.quantile(samples, quantiles)
            spread = (samples.max() - samples.min()) / (self.K + 1) + 1e-6
            sigmas = np.full(self.K, spread)
            target_entropy = 0.5 * np.log(self.K)
            for _ in range(self.max_iter):
                d2 = (samples[None, :] - centers[:, None]) ** 2
                unnorm = np.exp(-d2 / (2 * sigmas[:, None] ** 2 + 1e-12))
                mu = unnorm / (unnorm.sum(axis=0, keepdims=True) + 1e-12)
                new_centers = (mu * samples[None, :]).sum(axis=1) / (mu.sum(axis=1) + 1e-12)
                entropy = np.mean(-(mu * np.log(mu + 1e-12)).sum(axis=0))
                grad = target_entropy - entropy
                new_sigmas = np.clip(sigmas + self.lr * grad * sigmas, self.sigma_min, None)
                shift = np.max(np.abs(new_centers - centers))
                centers, sigmas = new_centers, new_sigmas
                if shift < self.tol:
                    break
            return {"centers": centers, "sigmas": sigmas}

    def fit(self, coeff_stack):
        """coeff_stack: (n, num_samples) pooled scalar coefficients per coefficient index."""
        n = coeff_stack.shape[0]
        params = [self.fit_one_coefficient(coeff_stack[k]) for k in range(n)]
        if self.K == 2:
            quantizer = SigmoidFuzzyQuantizer(n)
            quantizer.c = np.array([p["c"] for p in params])
            quantizer.tau = np.array([p["tau"] for p in params])
            return quantizer
        else:
            part = GaussianRBFPartition(n, self.K)
            part.centers = np.stack([p["centers"] for p in params])
            part.sigmas = np.stack([p["sigmas"] for p in params])
            return part
