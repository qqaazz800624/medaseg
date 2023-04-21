#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="manafaln",
    version="0.5.1",
    author=[
        "Pochuan Wang",
        "Tianyu Hwang"
    ],
    author_email=[
        "d08922016@csie.ntu.edu.tw",
        "tyhwang@ncts.ntu.edu.tw"
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "monai>=1.1.0",
        "pytorch-lightning<2.0.0",
        "ruamel.yaml>=0.17.21",
        "Pillow>=9.5.0",
        "scipy>=1.10.1",
        "scikit-learn>=1.2.2",
        "scikit-image>=0.20.0",
        "pandas>=2.0.0",
        "tensorboard>=2.12.2"
    ],
    tests_require=["pytest"],
    python_requires=">=3.8"
)
