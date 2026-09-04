"""
Section 8 — End-to-end pipeline (Algorithm 8).

`load_dataset` expects a folder structured as `root/<class_name>/<image
files>` (this is the layout you'd get from unzipping KTH-TIPS, Outex or
Kylberg into per-class subfolders). `run_experiment` performs the full
pipeline for one (descriptor, mode) combination:
  descriptor in {"LPQ", "LPQ6", "MP-LPQ"}
  mode in {"crisp_hist", "fuzzy_hist", "fuzzy_fv"}
"""

import os
from glob import glob

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

try:
    from PIL import Image
except ImportError:
    Image = None

from .crisp import CONFIG_MAP, extract_crisp_descriptor
from .fuzzy_quantizers import AdaptiveFuzzyPartitionLearner
from .fuzzy_descriptor import extract_fuzzy_descriptor
from .stft import stft_coefficients
from .fisher_vector import train_fv_gmm, fisher_vector_encode


def load_dataset(root_dir, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
                  resize=None, max_per_class=None):
    if Image is None:
        raise ImportError("Pillow is required for load_dataset (pip install pillow)")
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    images, labels = [], []
    for cls in classes:
        paths = []
        for ext in extensions:
            paths += glob(os.path.join(root_dir, cls, f"*{ext}"))
        paths = sorted(paths)
        if max_per_class is not None:
            paths = paths[:max_per_class]
        for p in paths:
            img = Image.open(p).convert("L")
            if resize is not None:
                img = img.resize(resize)
            images.append(np.array(img, dtype=np.float64))
            labels.append(cls)
    return images, labels, classes


def fit_fuzzy_quantizers(images, descriptor="LPQ", K=2, learner_kwargs=None):
    """Pool STFT coefficients across training images and learn one fuzzy partition
    per coefficient index, per frequency-point configuration (Algorithm 3)."""
    configs = CONFIG_MAP[descriptor]
    learner_kwargs = learner_kwargs or {}
    quantizers = []
    for cfg in configs:
        pooled = []
        for img in images:
            coeffs = stft_coefficients(img, **cfg)
            pooled.append(coeffs.reshape(coeffs.shape[0], -1))
        pooled = np.concatenate(pooled, axis=1)
        learner = AdaptiveFuzzyPartitionLearner(K=K, **learner_kwargs)
        quantizers.append(learner.fit(pooled))
    return configs, quantizers


def batch_extract(images, mode, descriptor="LPQ", grid=(4, 4), quantizers=None, gmm=None):
    """
    mode: 'crisp_hist' | 'fuzzy_hist' | 'fuzzy_fv'
    Returns (feature_matrix, local_descriptors_per_image); the latter is only
    populated (as a list) when mode == 'fuzzy_fv' and gmm is None (pooling stage).
    """
    configs = CONFIG_MAP[descriptor]
    feats, locals_ = [], []
    for img in images:
        if mode == "crisp_hist":
            feats.append(extract_crisp_descriptor(img, configs, grid=grid))
        elif mode in ("fuzzy_hist", "fuzzy_fv"):
            hist, local = extract_fuzzy_descriptor(img, configs, quantizers, grid=grid)
            if mode == "fuzzy_hist":
                feats.append(hist)
            else:
                locals_.append(local)
                if gmm is not None:
                    feats.append(fisher_vector_encode(local, gmm))
        else:
            raise ValueError(f"unknown mode {mode}")
    if mode == "fuzzy_fv" and gmm is None:
        return None, locals_
    return np.stack(feats), locals_


def run_experiment(train_images, train_labels, test_images, test_labels,
                    descriptor="LPQ", mode="crisp_hist", K=2, grid=(4, 4),
                    n_gaussians=32, seed=0, svm_kernel="rbf", svm_C=10.0):
    """Full Algorithm 8 pipeline for one (descriptor, mode) combination."""
    quantizers, gmm = None, None

    if mode in ("fuzzy_hist", "fuzzy_fv"):
        _, quantizers = fit_fuzzy_quantizers(train_images, descriptor=descriptor, K=K)

    if mode == "crisp_hist":
        Xtr, _ = batch_extract(train_images, "crisp_hist", descriptor, grid)
        Xte, _ = batch_extract(test_images, "crisp_hist", descriptor, grid)
    elif mode == "fuzzy_hist":
        Xtr, _ = batch_extract(train_images, "fuzzy_hist", descriptor, grid, quantizers)
        Xte, _ = batch_extract(test_images, "fuzzy_hist", descriptor, grid, quantizers)
    elif mode == "fuzzy_fv":
        _, local_tr = batch_extract(train_images, "fuzzy_fv", descriptor, grid, quantizers, gmm=None)
        pooled = np.concatenate(local_tr, axis=0)
        if pooled.shape[0] > 20000:
            idx = np.random.RandomState(seed).choice(pooled.shape[0], 20000, replace=False)
            pooled = pooled[idx]
        gmm = train_fv_gmm(pooled, n_components=n_gaussians, seed=seed)
        Xtr, _ = batch_extract(train_images, "fuzzy_fv", descriptor, grid, quantizers, gmm=gmm)
        Xte, _ = batch_extract(test_images, "fuzzy_fv", descriptor, grid, quantizers, gmm=gmm)
    else:
        raise ValueError(mode)

    clf = SVC(kernel=svm_kernel, C=svm_C)
    clf.fit(Xtr, train_labels)
    preds = clf.predict(Xte)
    acc = accuracy_score(test_labels, preds)
    report = classification_report(test_labels, preds, zero_division=0)
    return acc, report, (Xtr, Xte)
