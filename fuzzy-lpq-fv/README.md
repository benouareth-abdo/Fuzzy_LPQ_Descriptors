# Fuzzy LPQ / LPQ6 / MP-LPQ with Fisher Vector Embedding

Reference implementation accompanying the paper:

> **Fuzzy Local Phase Quantization with Fisher Vector Embedding for
> Blur-Insensitive Texture Classification**
> *(manuscript to be submitted to Expert Systems With Applications)*

This repository implements, end-to-end, the ideas of the paper:

1. **STFT phase coefficient extraction** (Algorithm 1)
2. **Crisp LPQ, LPQ6, MP-LPQ** histogram descriptors (Algorithm 2)
3. **Fuzzy quantization**: sigmoid (K=2) and Gaussian RBF (K>2) membership
   functions (Eqs. 3–4)
4. **Adaptive fuzzy partition learning**, an EM-style gradient-on-center
   procedure (Algorithm 3)
5. **Fuzzy soft-histogram extraction** via the Kronecker-product voting
   trick (Algorithms 4–5)
6. **Blur degradation bank** matching MATLAB's `fspecial` (Gaussian /
   motion / disk)
7. **Fisher Vector encoding** over a diagonal-covariance GMM
   (Algorithms 6–7)
8. **End-to-end classification pipeline** (Algorithm 8): dataset loading
   → descriptor extraction → SVM training/evaluation

A **synthetic-texture demo** reproduces the paper's comparison table
(LPQ / LPQ6 / MP-LPQ, crisp vs. fuzzy vs. fuzzy+FV), and instructions are
included for plugging in KTH-TIPS / Outex / Kylberg.

Everything is pure NumPy/SciPy/scikit-learn — no GPU required.

## Repository layout

```
fuzzy-lpq-fv/
├── fuzzy_lpq_fv/
│   ├── __init__.py           # package exports
│   ├── stft.py                # Sec. 1 — STFT phase coefficient extraction
│   ├── crisp.py                # Sec. 2 — crisp LPQ / LPQ6 / MP-LPQ
│   ├── fuzzy_quantizers.py     # Sec. 3-4 — fuzzy membership + adaptive partition learning
│   ├── fuzzy_descriptor.py     # Sec. 5 — fuzzy soft-histogram extraction
│   ├── blur_bank.py            # Sec. 6 — blur degradation bank (fspecial-equivalent)
│   ├── fisher_vector.py        # Sec. 7 — Fisher Vector encoding (GMM-based)
│   └── pipeline.py             # Sec. 8 — end-to-end pipeline (Algorithm 8)
├── demo.py                     # synthetic-texture demo, runs instantly
├── run_real_dataset.py         # template for KTH-TIPS / Outex / Kylberg
├── Fuzzy_LPQ_Descriptors.ipynb   # original Colab notebook (reference)
├── requirements.txt
├── setup.py
└── LICENSE
```

## Installation

```bash
git clone <this-repo-url>
cd fuzzy-lpq-fv
pip install -r requirements.txt
# optional, to `import fuzzy_lpq_fv` from anywhere:
pip install -e .
```

## Quick start

Run the synthetic-texture sanity check (no dataset required):

```bash
python demo.py
```

This fabricates three toy texture classes, applies a Gaussian blur to
the test partition, and prints a comparison table across all three
descriptors (LPQ, LPQ+, LPQ++) and all three modes (crisp histogram,
fuzzy histogram, fuzzy + Fisher Vector).

## Running on real datasets

1. Unzip your dataset so each class is its own subfolder:
   ```
   <root>/<class_name>/*.png
   ```
2. Edit `DATASET_ROOT` in `run_real_dataset.py` and run it, or call the
   pipeline directly:

```python
from sklearn.model_selection import train_test_split
from fuzzy_lpq_fv import load_dataset, apply_blur, BLUR_BANK, run_experiment

images, labels, classes = load_dataset(
    "/path/to/textures/kylberg", resize=(128, 128), max_per_class=40,
)

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3, stratify=labels, random_state=0,
)

# Optionally degrade the test set with a blur kernel to test blur-robustness:
test_images = [apply_blur(im, BLUR_BANK["gaussian_s2"]()) for im in test_images]

acc, report, _ = run_experiment(
    train_images, train_labels, test_images, test_labels,
    descriptor="MP-LPQ", mode="fuzzy_fv", grid=(4, 4), n_gaussians=32,
)
print(acc)
print(report)
```

Standard evaluation protocol used in the paper: 5-fold cross-validation
across seven blur conditions (Gaussian, motion, disk) on the KTH-TIPS,
Outex, and Kylberg benchmark datasets, across four evaluation modes
(crisp/fuzzy × histogram/Fisher Vector), reporting mean ± std per blur
condition.

## Notes

- **Speed**: `stft_coefficients` uses `fftconvolve`, fast enough for
  moderate image sizes (up to a few hundred pixels per side) on CPU;
  for larger-scale runs consider batching images and/or moving the STFT
  step to a GPU (e.g. via CuPy or `torch.fft`).
- **Fuzziness parameter**: the target-entropy heuristics inside
  `AdaptiveFuzzyPartitionLearner` (`target_entropy = 0.85` for K=2,
  `target_entropy = 0.5*log(K)` for K>2) control how "soft" the learned
  partition is; lowering them moves the learned quantizer closer to the
  crisp descriptor (tau_k → 0), as discussed in Section 4 and the
  ablation of Section 8.2 of the paper.
- **Fisher Vector dimensionality**: the FV has dimension `2*M*d` where
  `M` is the number of Gaussians and `d` the fuzzy local descriptor
  dimension (`n` per configuration, summed across the `P` configurations
  for MP-LPQ); reduce `n_gaussians` if you hit memory limits with large
  datasets.

## Related work

This work builds on the blur-insensitive texture descriptor literature,
in particular:

- V. Ojansivu, J. Heikkilä. *Blur Insensitive Texture Classification
  Using Local Phase Quantization*. ICISP 2008.
- Xiao et al. *Local phase quantization plus: A principled method for
  embedding local phase quantization into Fisher vector for blurred
  image recognition*. Information Sciences, 2017.
- Zhu et al. *LPQ++: A discriminative blur-insensitive textural
  descriptor with spatial-channel interaction*. Information Sciences,
  2021.

## Citation

This paper is not yet submitted/published. A citation entry (BibTeX)
will be added here once it is accepted for publication in Expert
Systems With Applications.

## License

MIT — see [LICENSE](LICENSE).
