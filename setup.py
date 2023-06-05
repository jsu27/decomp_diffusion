from setuptools import setup

setup(
    name="decomp_diffusion",
    packages=[
        "decomp_diffusion",
        "decomp_diffusion.diffusion",
        "decomp_diffusion.model",
        "decomp_diffusion.util"
    ],
    install_requires=[
        # "Pillow",
        # "attrs",
        "torch",
        # "filelock",
        # "requests",
        "tqdm",
        # "ftfy",
        # "regex",
        "numpy",
        "blobfile",
        "torchvision",
        # "diffuser",
        # "transformers",
        # "filelock",
        # "requests",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "scipy",
        "imageio",
        "ema_pytorch" # optional
    ],
    author="jsu27",
    version='1.0',
)
