"""
Fuzzy LPQ / LPQ6 / MP-LPQ with Fisher Vector Embedding
------------------------------------------------------
Reference implementation accompanying the paper:

    "Fuzzy Local Phase Quantization with Fisher Vector Embedding for
    Blur-Insensitive Texture Classification"
    (submitted to Expert Systems With Applications)

This package implements, end-to-end:

1. STFT phase coefficient extraction (Algorithm 1)
2. Crisp LPQ, LPQ6, MP-LPQ histogram descriptors (Algorithm 2)
3. Fuzzy quantization: sigmoid (K=2) and Gaussian RBF (K>2) membership
   functions (Eqs. 3-4)
4. Adaptive fuzzy partition learning (Algorithm 3)
5. Fuzzy soft-histogram extraction via the Kronecker-product voting
   trick (Algorithms 4-5)
6. A blur degradation bank matching MATLAB's `fspecial`
7. Fisher Vector encoding over a diagonal-covariance GMM (Algorithms 6-7)
8. An end-to-end classification pipeline (Algorithm 8)
"""

from .stft import stft_coefficients
from .crisp import (
    LPQ_CONFIG,
    LPQ6_CONFIG,
    MP_LPQ_CONFIGS,
    CONFIG_MAP,
    extract_crisp_descriptor,
)
from .fuzzy_quantizers import (
    SigmoidFuzzyQuantizer,
    GaussianRBFPartition,
    AdaptiveFuzzyPartitionLearner,
)
from .fuzzy_descriptor import extract_fuzzy_descriptor
from .blur_bank import (
    fspecial_gaussian,
    fspecial_disk,
    fspecial_motion,
    apply_blur,
    BLUR_BANK,
)
from .fisher_vector import train_fv_gmm, fisher_vector_encode
from .pipeline import (
    load_dataset,
    fit_fuzzy_quantizers,
    batch_extract,
    run_experiment,
)

__version__ = "0.1.0"

__all__ = [
    "stft_coefficients",
    "LPQ_CONFIG",
    "LPQ6_CONFIG",
    "MP_LPQ_CONFIGS",
    "CONFIG_MAP",
    "extract_crisp_descriptor",
    "SigmoidFuzzyQuantizer",
    "GaussianRBFPartition",
    "AdaptiveFuzzyPartitionLearner",
    "extract_fuzzy_descriptor",
    "fspecial_gaussian",
    "fspecial_disk",
    "fspecial_motion",
    "apply_blur",
    "BLUR_BANK",
    "train_fv_gmm",
    "fisher_vector_encode",
    "load_dataset",
    "fit_fuzzy_quantizers",
    "batch_extract",
    "run_experiment",
]
