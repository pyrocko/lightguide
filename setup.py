import setuptools
from setuptools_rust import Binding, RustExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    zip_safe=False,
    rust_extensions=[
        RustExtension(
            "lightguide.afk_filter",
            path="Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    packages=["lightguide"],
)
