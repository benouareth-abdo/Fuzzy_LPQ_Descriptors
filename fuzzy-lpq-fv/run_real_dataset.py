"""
Example: running on real datasets (KTH-TIPS / Outex / Kylberg).

1. Unzip your dataset so that each class is its own subfolder:
       <root>/<class_name>/*.png

2. Load it, split into train/test (e.g. standard splits from
   Xiao et al. 2017 / Zhu et al. 2021, or a random stratified split for
   a quick check), optionally apply the blur bank to the test
   partition, then call `run_experiment`.

Edit DATASET_ROOT below and uncomment the block to run.
"""

from sklearn.model_selection import train_test_split

from fuzzy_lpq_fv import load_dataset, apply_blur, BLUR_BANK, run_experiment

DATASET_ROOT = "/path/to/textures/kylberg"


def main():
    images, labels, classes = load_dataset(
        DATASET_ROOT, resize=(128, 128), max_per_class=40,
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


if __name__ == "__main__":
    main()
