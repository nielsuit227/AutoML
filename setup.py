#  Copyright (c) 2022 by Amplo.

import subprocess

import setuptools

if __name__ == "__main__":
    version = (
        subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )
    if "." not in version:
        raise ValueError(f"Invalid version: {version}")

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    with open("requirements.txt") as f:
        required = f.read().splitlines()

    setuptools.setup(
        name="Amplo",
        version=version,
        author="Amplo AG",
        author_email="info@amplo.ch",
        description="Fully automated end to end machine learning pipeline",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/nielsuit227/AutoML",
        packages=setuptools.find_packages(),
        package_data={"Amplo": ["VERSION"]},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.9",
        install_requires=required,
    )
