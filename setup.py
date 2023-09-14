#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="manafaln",
    version="0.5.2",
    author=["Pochuan Wang", "Tianyu Hwang"],
    author_email=["d08922016@csie.ntu.edu.tw", "tyhwang@ncts.ntu.edu.tw"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "monai>=1.1.0",
        "pytorch-lightning<2.0.0",
        "ruamel.yaml>=0.17.21",
        "scikit-image>=0.20.0",
        "pandas>=2.0.0",
        "tensorboard>=2.12.2",
        "matplotlib>=3.7.1",
    ],
    tests_require=["pytest"],
    python_requires=">=3.8",
)
