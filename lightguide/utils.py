from __future__ import annotations

import time
from functools import wraps
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Callable, Union

import numpy as np
import requests

PathStr = Union[Path, str]


class ExampleData:
    VSPData = "https://data.pyrocko.org/testing/lightguide/das-data.npy"
    EQData = "https://data.pyrocko.org/testing/lightguide/data-DAS-gfz2020wswf.npy"


class AttrDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def download_http(url: str, target: Path | SpooledTemporaryFile) -> None:
    req = requests.get(url)
    req.raise_for_status()
    total_size = int(req.headers.get("Content-Length", 0))
    n_bytes = 0

    if isinstance(target, Path):
        if target.exists():
            raise OSError(f"File {target} already exists")

        def file_writer(data: bytes):
            with target.open("ab") as f:
                f.write(data)

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
        print(f"{func.__qualname__} took {time.time() - t:.4f} s")
        return ret

    return wrapper
