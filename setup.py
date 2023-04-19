#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="manafaln",
    version="0.5.1",
    author=["Pochuan Wang", "Tianyu Hwang"],
    author_email=["d08922016@csie.ntu.edu.tw", "tyhwang@ncts.ntu.edu.tw"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "monai>=1.0.0",
        "pytorch-lightning<2.0.0",
        "ruamel.yaml"
    ],
    tests_require=["pytest"],
    python_requires=">=3.8"
)
