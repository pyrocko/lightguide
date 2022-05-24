import setuptools
from setuptools_rust import Binding, RustExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lightguide",
    version="0.1.0",
    author="Marius Paul Isken",
    author_email="mi@gfz-potsdam.de",
    description="DAS Tools for Pyrocko",
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyrocko/lightguide",
    project_urls={
        "Bug Tracker": "https://github.com/pyrocko/lightguide/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    rust_extensions=[
        RustExtension(
            "lightguide.orafk_filter",
            path="Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    package_data={"lightguide": ["data/*.npy"]},
    packages=["lightguide"],
    python_requires=">=3.8",
)
