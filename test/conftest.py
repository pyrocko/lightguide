import pytest
from lightguide import gf

km = 1e3


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False)


@pytest.fixture
def show_plot(pytestconfig) -> bool:
    return pytestconfig.getoption("plot")


@pytest.fixture
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

        traces = engine.process_fiber(source, fiber)
        return traces

    return get_data
