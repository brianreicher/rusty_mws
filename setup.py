from setuptools import setup

setup(
    name="rusty_mws",
    version="0.1.0",
    description="An optimized Rust-based mutex watershed",
    long_description=open("README.md").read(),
    author="Brian Kyle Reicher",
    author_email="reicher.b@northeastern.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.6",
    install_requires=[
            "numpy",
            "pymongo",
            "scikit-image",
            "mwatershed @ git+https://github.com/pattonw/mwatershed",
            "funlib.geometry @ git+https://github.com/funkelab/funlib.geometry@976dc2c#egg=funlib.geometry",
            "gunpowder @ git+https://github.com/funkey/gunpowder@68dffe1#egg=gunpowder",
            "funlib.persistence @ git+https://github.com/funkelab/funlib.persistence.git",
            "daisy",
            "funlib.segment @ git+https://github.com/funkelab/funlib.segment.git"
    ]
)
