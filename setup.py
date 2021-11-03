#!/usr/bin/env python

from setuptools import setup

setup(
    name="manafaln",
    version="0.1.0-dev",
    author="Pochuan Wang",
    author_email="d08922016@csie.ntu.edu.tw",
    packages=["manafaln"],
    include_package_data=True,
    install_requires=[
        "monai>=0.6.0",
        "pytorch-lightning>=1.4.9"
    ]
)
