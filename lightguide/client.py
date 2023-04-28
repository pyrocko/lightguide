from datetime import datetime

import numpy as np

from pathlib import Path
from lightguide.blast import Blast
from lightguide.utils import cache_dir, download_http


class ExampleData:
    VSPDataUrl = "https://data.pyrocko.org/testing/lightguide/VSP-DAS-G1-120.mseed"
    DataUrl = "https://data.pyrocko.org/testing/lightguide/das-data.npy"
    EQDataUrl = "https://data.pyrocko.org/testing/lightguide/data-DAS-gfz2020wswf.npy"
    IceQDataUrl = "https://github.com/sachalapins/DAS-N2N/raw/main/data/BPT1_UTC_20200117_044207.903.mseed"
    # MarkersUrl = "https://data.pyrocko.org/testing/lightguide/markers_VSP-DAS-G1-120.txt"

    @classmethod
    def earthquake(cls) -> Blast:
        file = cache_dir() / "data-DAS-gfz2020wswf.npy"
        if not file.exists():
            download_http(cls.EQDataUrl, file)

        return Blast(
            np.load(file),
            start_time=datetime.fromisoformat("2020-11-19T09:27:08.190+00:00"),
            sampling_rate=200.0,
            channel_spacing=1.0,
        )

    @classmethod
    def vsp_shot(cls) -> Blast:
        file = cache_dir() / "vsp-das-g1-120.mseed"
        if not file.exists():
            download_http(cls.VSPDataUrl, file)

        return Blast.from_miniseed(file, channel_spacing=1.0)

    @classmethod
    def icequake(cls) -> Blast:
        file = cache_dir() / "BPT1_UTC_20200117_044207.903.mseed"
        if not file.exists():
            download_http(cls.IceQDataUrl, file)

        return Blast.from_miniseed(file, channel_spacing=1.0)

    @classmethod
    def markerfile(cls) -> Blast:
        file = Path("../examples/markers_VSP-DAS-G1-120.txt")
        if not file.exists():
            print(
                "Markerfile has not yet been created.\n \
                to create a markerfile please see EXAMPLE 3"
            )
            # file = cache_dir() / "markers_VSP-DAS-G1-120.txt"
            # download_http(cls.MarkersUrl, file)

        return file
