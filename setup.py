from setuptools import find_packages, setup

with open("requirements_gh_action.txt") as f:
    required = f.read().splitlines()

setup(
    name="overflow",
    packages=find_packages(),
    version="1.0.0",
    license="MIT",
    description="OverFlow: Putting flows on top of neural transducers for better TTS",
    author="Shivam Mehta",
    author_email="shivammehta25@gmail.com",
    url="https://shivammehta25.github.io/OverFlow/",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "audio",
        "generative modelling",
        "speech synthesis",
    ],
    install_requires=required,
    entry_points={"console_scripts": ["overflow = src.cli:cli"]},
)
