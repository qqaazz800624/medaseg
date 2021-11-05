#!/usr/bin/env python

from setuptools import setup

packages = [
    "manafaln",
    "manafaln.common",
    "manafaln.data",
    "manafaln.workflow",
    "manafaln.utils"
]

setup(
    name="manafaln",
    version="0.1.0.dev0",
    author="Pochuan Wang",
    author_email="d08922016@csie.ntu.edu.tw",
    packages=packages,
    include_package_data=True,
    install_requires=[
        "monai>=0.6.0",
        "pytorch-lightning>=1.4.9"
    ],
    python_requires=">=3.8"
)
