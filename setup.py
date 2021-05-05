import setuptools
# import pytorch_sfid

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="pytorch-sfid",
    version="0.0.1",  # pytorch_sfid.__version__,
    author="Even M. Nordhagen",
    author_email="not.an.address@yes.com",
    description=(
        "Wrapper around the pytorch-fid package to compute Frechet Inception"
        + "Distance (FID) using PyTorch in-memory given tensors of images."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evenmn/pytorch-sfid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.5",
    install_requires=["torch", "pytorch-fid-wrapper"],
)
