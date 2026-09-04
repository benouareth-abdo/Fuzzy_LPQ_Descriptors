"""
Synthetic-texture demo (sanity check, runs instantly).

Fabricates a few toy texture classes so the whole pipeline can be
exercised without needing to download a dataset first, and reproduces
the shape of the paper's comparison table (Table 1): LPQ / LPQ6 / MP-LPQ,
crisp vs. fuzzy vs. fuzzy+FV.

For real experiments on KTH-TIPS / Outex / Kylberg, see the "Real
datasets" section of the README.
"""

import numpy as np

from fuzzy_lpq_fv import BLUR_BANK, apply_blur, run_experiment


def make_texture(kind, size=48, seed=None):
    rng = np.random.RandomState(seed)
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    if kind == "stripes":
        img = 128 + 100 * np.sin(2 * np.pi * xs / 6)
    elif kind == "checker":
        img = 128 + 100 * np.sign(np.sin(2 * np.pi * xs / 8) * np.sin(2 * np.pi * ys / 8))
    elif kind == "waves":
        img = 128 + 100 * np.sin(2 * np.pi * xs / 5) * np.cos(2 * np.pi * ys / 5)
    else:
        raise ValueError(kind)
    noise = rng.randn(size, size) * 10
    return np.clip(img + noise, 0, 255)


def main():
    classes = ["stripes", "checker", "waves"]
    train_images, train_labels, test_images, test_labels = [], [], [], []
    for c in classes:
        for i in range(15):
            train_images.append(make_texture(c, seed=i))
            train_labels.append(c)
        for i in range(8):
            img = make_texture(c, seed=1000 + i)
            img = apply_blur(img, BLUR_BANK["gaussian_s1"]())  # blur-degraded test set
            test_images.append(img)
            test_labels.append(c)

    print(f"{len(train_images)} training images, {len(test_images)} blur-degraded "
          f"test images, {len(classes)} classes")

    results = {}
    for descriptor in ["LPQ", "LPQ6", "MP-LPQ"]:
        for mode in ["crisp_hist", "fuzzy_hist", "fuzzy_fv"]:
            acc, report, _ = run_experiment(
                train_images, train_labels, test_images, test_labels,
                descriptor=descriptor, mode=mode, grid=(2, 2), n_gaussians=8,
            )
            results[(descriptor, mode)] = acc
            print(f"{descriptor:6s} | {mode:11s} | accuracy = {acc:.3f}")

    print()
    print(f"{'Descriptor':10s} {'crisp':>8s} {'fuzzy':>8s} {'fuzzy+FV':>10s}")
    for descriptor in ["LPQ", "LPQ6", "MP-LPQ"]:
        row = results
        print(f"{descriptor:10s} {row[(descriptor,'crisp_hist')]:8.3f} "
              f"{row[(descriptor,'fuzzy_hist')]:8.3f} {row[(descriptor,'fuzzy_fv')]:10.3f}")


if __name__ == "__main__":
    main()
