from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt


@lru_cache
def triangular_taper_python(size: int, plateau: int) -> npt.NDArray:

    if plateau > size:
        raise ValueError("Plateau cannot be larger than size.")
    if size % 2 or plateau % 2:
        raise ValueError("Size and plateau have to be even.")

    ramp_size = int((size - plateau) / 2)
    ramp = np.linspace(0.0, 1.0, ramp_size + 2)[1:-1]
    window = np.ones(size)
    window[:ramp_size] = ramp
    window[size - ramp_size :] = ramp[::-1]
    return window * window[:, np.newaxis]


def afk_filter_python(
    data: npt.NDArray,
    window_size: int = 32,
    overlap: int = 14,
    exponent: float = 0.3,
    normalize_power: bool = False,
) -> npt.NDArray:
    if np.log2(window_size) % 1.0 or window_size < 4:
        raise ValueError("window_size has to be pow(2) and > 4.")
    if overlap > window_size / 2 - 1:
        raise ValueError("Overlap is too large. Maximum overlap: window_size / 2 - 1.")

    window_stride = window_size - overlap
    window_non_overlap = window_size - 2 * overlap

    npx_x, npx_y = data.shape
    nwin_x = npx_x // window_stride
    nwin_y = npx_y // window_stride
    if nwin_x % 1 or nwin_y % 1:
        raise ValueError("Padding does not match desired data shape")

    filtered_data = np.zeros_like(data)
    taper = triangular_taper_python(window_size, window_non_overlap)

    for iwin_x in range(nwin_x):
        px_x = iwin_x * window_stride
        slice_x = slice(px_x, px_x + window_size)
        for iwin_y in range(nwin_y):
            px_y = iwin_y * window_stride
            slice_y = slice(px_y, px_y + window_size)

            window_data = data[slice_x, slice_y]
            window_fft = np.fft.rfft2(window_data)

            power_spec = np.abs(window_fft)
            # power_spec = signal.medfilt2d(power_spec, kernel_size=3)
            if normalize_power:
                power_spec /= power_spec.max()

            weights = power_spec**exponent
            window_fft *= weights
            # window_fft /= weights.sum()
            window_flt = np.fft.irfft2(window_fft)
            # window_flt /= weights.max()
            taper_this = taper[: window_flt.shape[0], : window_flt.shape[1]]
            window_flt *= taper_this
            filtered_data[
                px_x : px_x + window_flt.shape[0],
                px_y : px_y + window_flt.shape[1],
            ] += window_flt

    return filtered_data
