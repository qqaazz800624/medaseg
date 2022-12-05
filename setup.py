#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="manafaln",
    version="0.4.0",
    author="Pochuan Wang",
    author_email="d08922016@csie.ntu.edu.tw",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "monai>=1.0.0",
        "pytorch-lightning>=1.7.0",
        "ruamel.yaml"
    ],
    tests_require=["pytest"],
    python_requires=">=3.8"
)
