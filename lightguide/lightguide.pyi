from __future__ import annotations

import numpy.typing as npt
import numpy as np

def triangular_taper(
    window_size: tuple[int, int], plateau: tuple[int, int]
) -> npt.NDArray[np.float32]: ...
def afk_filter(
    data: npt.NDArray[np.float32],
    window_size: int,
    overlap: int,
    exponent: float,
    normalize_power: bool,
) -> npt.NDArray[np.float32]: ...
def afk_filter_rectangular(
    data: npt.NDArray[np.float32],
    window_size: tuple[int, int],
    overlap: tuple[int, int],
    exponent: float,
    normalize_power: bool,
) -> npt.NDArray[np.float32]: ...
