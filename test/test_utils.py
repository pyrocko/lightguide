from pathlib import Path
from lightguide.utils import download_http, download_numpy, ExampleData
import numpy as np

URLS = [url for k, url in ExampleData.__dict__.items() if not k.startswith("_")]


def test_downloads():
    for url in URLS:
        download_http(url, Path("/tmp/test"))


def test_download_numnpy():
    for url in URLS:
        res = download_numpy(url)
        assert isinstance(res, np.ndarray)


if __name__ == "__main__":
    test_download_numnpy()
