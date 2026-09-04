from setuptools import setup, find_packages

setup(
    name="fuzzy-lpq-fv",
    version="0.1.0",
    description=(
        "Fuzzy Local Phase Quantization with Fisher Vector Embedding "
        "for Blur-Insensitive Texture Classification"
    ),
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "pillow",
    ],
    python_requires=">=3.8",
)
