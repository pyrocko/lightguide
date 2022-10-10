from __future__ import annotations

import logging
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import requests
from pyrocko.trace import Trace
from scipy.signal import butter, lfilter

# create console handler
ch = logging.StreamHandler()
formatter = logging.Formatter("\x1b[80D\x1b[1A\x1b[K%(message)s")
ch.setFormatter(formatter)


class ExampleData:
    VSPData = "https://data.pyrocko.org/testing/lightguide/das-data.npy"
    EQData = "https://data.pyrocko.org/testing/lightguide/data-DAS-gfz2020wswf.npy"


class AttrDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def traces_to_numpy_and_meta(traces: list[Trace]) -> tuple[npt.NDArray, AttrDict]:
    """Geneare a numpy 2-D array from a list of traces

    Args:
        traces (list[Trace]): List of input traces

    Raises:
        ValueError: Raised when the traces have different lengths / start times.

    Returns:
        tuple[npt.NDArray, AttrDict]: The waveform data and meta information as dict.
    """
    if not traces:
        raise ValueError("No traces given")
    ntraces = len(traces)
    nsamples = set(tr.ydata.size for tr in traces)

    if len(nsamples) != 1:
        raise ValueError("Traces number of samples differ.")

    nsamples = nsamples.pop()

    data = np.zeros((ntraces, nsamples))
    meta = {}
    for itr, tr in enumerate(traces):
        data[itr, :] = tr.ydata
        meta = tr.meta

    return data, AttrDict(meta)


def butter_bandpass_filter(
    data: npt.NDArray,
    lowcut: float,
    highcut: float,
    sampling_rate: float,
    order: int = 4,
) -> npt.NDArray:
    """Butterworth bandpass filter for 2D time-spatial array.

    Args:
        data (npt.NDArray): Input data as [time, space]
        lowcut (float): Low-cut frequency in [Hz].
        highcut (float): High-cut frequency in [Hz].
        sampling_rate (float): Sampling rate along the time dimension.
        order (int, optional): Order of the filter. Defaults to 4.

    Returns:
        npt.NDArray: Filtered wavefield.
    """
    coeff_b, coeff_a = butter(
        order, (lowcut, highcut), btype="bandpass", fs=sampling_rate
    )
    y = lfilter(coeff_b, coeff_a, data, axis=0)
    return y


def download_http(url: str, target: Path | SpooledTemporaryFile):
    req = requests.get(url)
    req.raise_for_status()
    total_size = int(req.headers.get("Content-Length", 0))
    nbytes = 0

    if isinstance(target, Path):
        writer = target.write_bytes
    elif isinstance(target, SpooledTemporaryFile):
        writer = target.write
    else:
        raise TypeError("Bad target for download")

    for data in req.iter_content(chunk_size=4096):
        nbytes += len(data)
        print(f"\u001b[2KDownloading {url}: {nbytes}/{total_size} bytes", end="\r")

        writer(data)
    print(f"\u001b[2KDownloaded {url}")


def download_numpy(url: str) -> np.ndarray:
    file = SpooledTemporaryFile()
    download_http(url, target=file)
    file.flush()
    file.seek(0)
    return np.load(file)


def timeit(func: Callable) -> Callable:
    """A helper decorator to time function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        t = time.time()
        ret = func(*args, **kwargs)
        print(func.__qualname__, time.time() - t)
        return ret

    return wrapper
