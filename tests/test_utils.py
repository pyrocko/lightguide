from pathlib import Path

import numpy as np

from lightguide.utils import ExampleData, download_http, download_numpy

URLS = [url for k, url in ExampleData.__dict__.items() if not k.startswith("_")]


def test_downloads(tmp_path):
    for url in URLS:
        download_http(url, tmp_path / Path(url).name)


def test_download_numpy():
    for url in URLS:
        res = download_numpy(url)
        assert isinstance(res, np.ndarray)
