from __future__ import annotations

import numpy as np

def triangular_taper(
    window_size: tuple[int, int], plateau: tuple[int, int]
) -> np.ndarray[np.float32]: ...
def afk_filter(
    data: np.ndarray[np.float32],
    window_size: int,
    overlap: int,
    exponent: float,
    normalize_power: bool,
) -> np.ndarray[np.dtype[np.float32]]:
    """Adaptive frequency filter for 2D data in square window.

    The adaptive frequency filter (AFK) can be used to suppress incoherent noise
    in DAS data and other spatially coherent data sets.

    Args:
        data (np.ndarray[np.float32]): Input data, `MxN`.
        window_size (int): Square window size.
        overlap (int): Overlap of neighboring windows.
        exponent (float): Filter strength, between 0.0 and 1.0.
        normalize_power (bool): Normalize the power.

    Returns:
        np.ndarray[np.dtype[np.float32]]: Filtered data.
    """

def afk_filter_rectangular(
    data: np.ndarray[np.dtype[np.float32]],
    window_size: tuple[int, int],
    overlap: tuple[int, int],
    exponent: float,
    normalize_power: bool,
) -> np.ndarray[np.dtype[np.float32]]:
    """Adaptive frequency filter for 2D data in rectangular window.

    The adaptive frequency filter (AFK) can be used to suppress incoherent noise
    in DAS data and other spatially coherent data sets.

    Args:
        data (np.ndarray[np.float32]): Input data, `MxN`.
        window_size (tuple[int, int]): Square window size.
        overlap (tuple[int, int]): Overlap of neighboring windows.
        exponent (float): Filter strength, between 0.0 and 1.0.
        normalize_power (bool): Normalize the power.

    Returns:
        np.ndarray[np.dtype[np.float32]]: Filtered data.
    """
