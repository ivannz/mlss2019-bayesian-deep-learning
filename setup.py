from distutils.core import setup

setup(
    name="mlss2019bdl",
    version="0.2",
    description="""MLSS2019 Tutorial on Bayesian Active Learning""",
    license="MIT License",
    author="Ivan Nazarov, Yarin Gal",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=[
        "mlss2019bdl",
        "mlss2019bdl.bdl",
    ],
    install_requires=[
        "numpy",
        "tqdm",
        "matplotlib",
        "torch",
        "torchvision",
    ]
)
