# Lightguide

*Tools for distributed acoustic sensing and modelling.*

Lightguide is a package for handling, filtering and modelling distributed acoustic sensing (DAS) data. The package interfaces handling and processing routines of DAS data to the [Pyrocko framework](https://pyrocko.org). Through Pyrocko's I/O engine :rocket: lightguide supports handling the following DAS data formats:

- Silixa iDAS (TDMS data)
- ASN OptoDAS
- MiniSEED

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
from lightguide import orafk_filter

filtered_data = orafk_filter.afk_filter(
    data, window_size=32, overlap=14, exponent=0.0, normalize_power=False)
```

The filtering performance of the AFK filter, applied to an earthquake recording at an [ICDP](https://www.icdp-online.org/home/) borehole observatory in Germany. The data was recorded on a [Silixa](https://silixa.com/) iDAS v2. For more details see <https://doi.org/10.5880/GFZ.2.1.2022.006>.

<img src="https://user-images.githubusercontent.com/4992805/170084970-9484afe7-9b95-45a0-ac8e-aec56ddfb3ea.png" style="width: 700px;" />

*The figures show the performance of the AFK filter applied to noisy DAS data. (a) Raw data. (b) The filtered wave field using the AFK filter with exponent = 0.6, 0.8, 1.0, 32 x 32 sample window size and 15 samples overlap. (c) The normalized residual between raw and filtered data. (d) Normalized raw (black) waveform and waveforms filtered (colored) by different filter exponents, the shaded area marks the signal duration. (e) Power spectra of signal shown in (d; shaded duration), the green area covers the noise band used for estimating the reduction in spectral amplitude in dB. The data are neither tapered nor band-pass filtered, the images in (a-c) are not anti-aliased.*

## Citation

Lightguide can be cited as

> Isken, Marius Paul; Christopher, Wollin; Heimann, Sebastian; Dahm, Torsten (2022): Lightguide - Tools for distributed acoustic sensing.

Details of the adaptive frequency filter are published here

> Isken, Marius Paul; Vasyura-Bathke, Hannes; Dahm, Torsten; Heimann, Sebastian (2022): De-noising distributed acoustic sensing data using an adaptive frequency-wavenumber filter, Geophysical Journal International.

## Packaging

To package lightguit requires Rust and the maturin build tool. maturin can be installed from PyPI or packaged as well. This is the simplest and recommended way of installing from source:

```sh
pip install maturin
maturin build
```

## Contribution

Contribution and merge requests by the community are welcome!

## License

lightguide was written by Marius Paul Isken and is licensed under the GNU GENERAL PUBLIC LICENSE v3.
