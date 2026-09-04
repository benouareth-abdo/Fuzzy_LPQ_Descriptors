"""
Section 7 — Fisher Vector encoding (Algorithms 6-7).

A diagonal-covariance GMM is trained (offline, via EM) on pooled
per-pixel fuzzy descriptors from the training set (`train_fv_gmm`), then
each image's descriptor set is encoded into a mean/variance-deviation
Fisher Vector, power- and l2-normalized (`fisher_vector_encode`).
"""

import numpy as np
from sklearn.mixture import GaussianMixture


def train_fv_gmm(pooled_local_descriptors, n_components=32, seed=0):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=seed,
        reg_covar=1e-4,
        max_iter=100,
    )
    gmm.fit(pooled_local_descriptors)
    return gmm


def fisher_vector_encode(local_descriptors, gmm):
    T, d = local_descriptors.shape
    means = gmm.means_
    covs = gmm.covariances_
    weights = gmm.weights_
    M = means.shape[0]
    gamma = gmm.predict_proba(local_descriptors)  # (T, M)
    G_X = np.zeros((M, d))
    G_S = np.zeros((M, d))
    for m in range(M):
        diff = (local_descriptors - means[m]) / np.sqrt(covs[m] + 1e-12)
        g = gamma[:, m][:, None]
        G_X[m] = (g * diff).sum(axis=0) / (T * np.sqrt(weights[m]) + 1e-12)
        G_S[m] = (g * (diff ** 2 - 1)).sum(axis=0) / (T * np.sqrt(2 * weights[m]) + 1e-12)
    fv = np.concatenate([G_X.ravel(), G_S.ravel()])
    fv = np.sign(fv) * np.sqrt(np.abs(fv))
    fv = fv / (np.linalg.norm(fv) + 1e-12)
    return fv
