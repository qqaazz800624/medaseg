#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="manafaln",
    version="0.7.0",
    author=["Pochuan Wang", "Tianyu Hwang"],
    author_email=["d08922016@csie.ntu.edu.tw", "tyhwang@ncts.ntu.edu.tw"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.13.1",
        "scikit-image>=0.23.2",
        "nibabel>=5.2.1",
        "pandas>=2.2.2",
        "matplotlib>=3.9.0",
        "ruamel.yaml>=0.18.6"
        "tqdm>=4.66.4",
        "tensorboard>=2.16.2",
        "torch>=2.3.0",
        "torchmerics>=1.4.0",
        "monai>=1.3.1",
        "lightning>=2.2.5"
    ],
    tests_require=["pytest"],
    python_requires=">=3.10",
)
