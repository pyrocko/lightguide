from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Callable, Union

import numpy as np
import requests

PathStr = Union[Path, str]


def download_http(url: str, target: Path | SpooledTemporaryFile) -> None:
    """Helper function for downloading data from HTTP.

    Args:
        url (str): URL to download data from.
        target (Path | SpooledTemporaryFile): File to dump data to.

    Raises:
        OSError: Raised when target exists.
        TypeError: Raised when target is ill defined.
    """
    req = requests.get(url)
    req.raise_for_status()
    total_size = int(req.headers.get("Content-Length", 0))
    n_bytes = 0

    if isinstance(target, Path):
        if target.exists():
            raise OSError(f"File {target} already exists")

        def file_writer(data: bytes) -> None:
            with target.open("ab") as file:
                file.write(data)

        writer = file_writer

    elif isinstance(target, SpooledTemporaryFile):
        writer = target.write
    else:
        raise TypeError(f"Bad target {target} for download")

    for data in req.iter_content(chunk_size=4096):
        n_bytes += len(data)
        print(f"\u001b[2KDownloading {url}: {n_bytes}/{total_size} bytes", end="\r")
        writer(data)

    print(f"\u001b[2KDownloaded {url}")


def download_numpy(url: str) -> np.ndarray:
    """Helper function downloading a .npy file

    Args:
        url (str): URL to NumPy file

    Returns:
        np.ndarray: Retrieved numpy array
    """
    file = SpooledTemporaryFile()
    download_http(url, target=file)
    file.flush()
    file.seek(0)
    return np.load(file)


def cache_dir() -> Path:
    cache = Path.home() / ".cache" / "lightguide"
    if not cache.exists():
        logging.info("Creating cache dir %s", cache)
        cache.mkdir(parents=True)
    return cache


def timeit(func: Callable) -> Callable:
    """A helper decorator to time function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        t = time.time()
        ret = func(*args, **kwargs)
        print(f"{func.__qualname__} took {time.time() - t:.4f} s")
        return ret

    return wrapper
