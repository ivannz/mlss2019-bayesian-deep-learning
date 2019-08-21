from distutils.core import setup

setup(
    name="mlss2019bal",
    version="0.2",
    description="""MLSS2019 Tutorial on Bayesian Active Learning""",
    license="MIT License",
    author="Ivan Nazarov, Yarin Gal",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=[
        "mlss2019bal",
    ],
    install_requires=[
        "numpy",
        "tqdm",
        "matplotlib",
        "torch",
        "torchvision",
    ]
)
