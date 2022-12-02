# Lightguide

*Tools for distributed acoustic sensing and modelling.*

![PyPI](https://img.shields.io/pypi/v/lightguide)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightguide)](https://pypi.org/project/lightguide)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![CI](https://github.com/pyrocko/lightguide/actions/workflows/build.yml/badge.svg)](https://github.com/pyrocko/lightguide/actions/workflows/build.yml)

Lightguide is a package for handling, filtering and modelling distributed acoustic sensing (DAS) data. The package interfaces handling and processing routines of DAS data to the [Pyrocko framework](https://pyrocko.org). Through Pyrocko's I/O engine :rocket: lightguide supports handling the following DAS data formats:

- Silixa iDAS (TDMS data)
- ASN OptoDAS
- MiniSEED

Numerical forward modelling of various dislocation sources in layered and homogeneous half-space towards DAS strain and strain-rate is employed through Pyrocko-Green's function package.

> The framework is still in Beta. Expect changes throughout all functions.

## Installation

Install the compiled Python wheels from PyPI:

```sh
pip install lightguide
```

## Usage

### Adaptive frequency filter

The adaptive frequency filter (AFK) can be used to suppress incoherent noise in DAS data sets.

```python
from lightguide import filters
from lightguide.utils import download_numpy, ExampleData


das_data = download_numpy(ExampleData.VSPData)

filtered_data = filters.afk_filter(
    das_data, window_size=32, overlap=15, exponent=0.8, normalize_power=False)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyrocko/lightguide/blob/master/examples/1-denoise-DAS-data.ipynb)

The filtering performance of the AFK filter, applied to an earthquake recording at an [ICDP](https://www.icdp-online.org/home/) borehole observatory in Germany. The data was recorded on a [Silixa](https://silixa.com/) iDAS v2. For more details see <https://doi.org/10.5880/GFZ.2.1.2022.006>.

![AFK Filter Performance](https://user-images.githubusercontent.com/4992805/170084970-9484afe7-9b95-45a0-ac8e-aec56ddfb3ea.png)

*The figures show the performance of the AFK filter applied to noisy DAS data. (a) Raw data. (b) The filtered wave field using the AFK filter with exponent = 0.6, 0.8, 1.0, 32 x 32 sample window size and 15 samples overlap. (c) The normalized residual between raw and filtered data. (d) Normalized raw (black) waveform and waveforms filtered (colored) by different filter exponents, the shaded area marks the signal duration. (e) Power spectra of signal shown in (d; shaded duration), the green area covers the noise band used for estimating the reduction in spectral amplitude in dB. The data are neither tapered nor band-pass filtered, the images in (a-c) are not anti-aliased.*

## Citation

Lightguide can be cited as:

> Marius Paul Isken, Sebastian Heimann, Christopher Wollin, Hannes Bathke, & Torsten Dahm. (2022). Lightguide - Seismological Tools for DAS data. Zenodo. <https://doi.org/10.5281/zenodo.6580579>

[![DOI](https://zenodo.org/badge/495774991.svg)](https://zenodo.org/badge/latestdoi/495774991)

Details of the adaptive frequency filter are published here:

> Marius Paul Isken, Hannes Vasyura-Bathke, Torsten Dahm, Sebastian Heimann, De-noising distributed acoustic sensing data using an adaptive frequency-wavenumber filter, Geophysical Journal International, 2022;, ggac229, <https://doi.org/10.1093/gji/ggac229>

[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fgji%2Fggac229-blue)](https://doi.org/10.1093/gji/ggac229)

## Packaging

To package lightguit requires Rust and the maturin build tool. maturin can be installed from PyPI or packaged as well. This is the simplest and recommended way of installing from source:

```sh
# Install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin and build
pip install maturin
maturin build
```

### Development

Local development through pip or maturin.

```sh
cd lightguide
pip3 install .[dev]
```

or

```sh
cd lightguide
maturin develop
```

The project utilizes pre-commit for clean commits, install the hooks via:

```sh
pre-commit install
```

## License

Contribution and merge requests by the community are welcome!

Lightguide was written by Marius Paul Isken and is licensed under the GNU GENERAL PUBLIC LICENSE v3.
