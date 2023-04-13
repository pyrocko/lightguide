from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from lightguide import gf, utils
from lightguide.blast import Blast

km = 1e3


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False)


@pytest.fixture
def show_plot(pytestconfig) -> bool:
    return pytestconfig.getoption("plot")


@pytest.fixture
def data_dir() -> Path:
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="module")
def syn_das_data():
    def get_data(**kwargs):
        import pyrocko.gf as pgf

        default_kwargs = dict(
            lat=0.0,
            lon=0.0,
            quantity="velocity",
            store_id="das_test",
            coordinates=((1100, 0), (2000, 1000), (2000, 3000)),
            channel_spacing=4.0,
        )

        default_kwargs.update(kwargs)
        fiber = gf.Fiber(**default_kwargs)
        engine = gf.LocalEngine(use_config=True)

        source = pgf.DCSource(lat=0.0, lon=0.0, depth=2 * km, strike=45.0, dip=30.0)

        return engine.process_fiber(source, fiber)

    return get_data


@pytest.fixture
def random_blast() -> Blast:
    rng = np.random.default_rng()
    data = rng.integers(-1000, 1000, size=(100, 5000)).astype(np.int32)
    return Blast(
        data=data,
        start_time=datetime.now(tz=timezone.utc),
        sampling_rate=100,
    )


@pytest.fixture
def data_eq(data_dir: Path) -> np.ndarray:
    file = data_dir / "data-DAS-gfz2020wswf.npy"
    if not file.exists():
        utils.download_http(utils.ExampleData.EQData, file)
    return np.load(file)


@pytest.fixture
def blast_eq(data_eq) -> Blast:
    return Blast(
        data_eq,
        start_time=datetime.fromisoformat("2020-11-19T09:27:08.190+00:00"),
        sampling_rate=200.0,
        channel_spacing=1.0,
    )
