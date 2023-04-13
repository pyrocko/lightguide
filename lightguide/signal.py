from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from scipy import signal


@lru_cache(maxsize=32, typed=True)
def decimation_coefficients(
    decimation_factor: int,
    order: int | None = None,
    filter_type: Literal["fir", "fir-remez", "cheby"] = "fir",
) -> tuple[np.ndarray, list[float], int]:
    """Get dicimation filter factors for lfilter.

    Args:
        decimation_factor (int): Decimation factor, must be >= 1.
        order (int | None, optional): Order of the filter. Defaults to None.
        filter_type (Literal["fir", "fir-remez", "cheby"], optional):
            The FIR filter type. Defaults to "fir".

    Returns:
        _type_: _description_
    """
    decimation_factor = int(decimation_factor)
    if decimation_factor <= 1:
        raise ValueError(f"Bad decimation factor {decimation_factor}")

    if filter_type == "fir":
        order = order or 30
        coeffs = signal.firwin(order + 1, 0.75 / decimation_factor, window="hamming")
        return coeffs, [1.0], order

    if filter_type == "fir-remez":
        order = order or 40 * decimation_factor + 1
        coeffs = signal.remez(
            order + 1,
            (0.0, 0.75 / decimation_factor, 1.0 / decimation_factor, 1.0),
            (1.0, 0.0),
            Hz=2,
            weight=(1, 50),
        )
        return coeffs, [1.0], order

    if filter_type == "cheby":
        order = order or 8
        coeffs, a = signal.cheby1(order, 0.05, 0.8 / decimation_factor)
        return coeffs, a, order

    raise ValueError(f"Unknown filter type {filter_type}")
