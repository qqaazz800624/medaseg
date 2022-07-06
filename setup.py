#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="manafaln",
    version="0.2.9",
    author="Pochuan Wang",
    author_email="d08922016@csie.ntu.edu.tw",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "monai>=0.8.0",
        "pytorch-lightning>=1.5.0",
        "ruamel.yaml"
    ],
    tests_require=["pytest"],
    python_requires=">=3.8"
)
