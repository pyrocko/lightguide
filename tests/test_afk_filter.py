from pathlib import Path

import numpy as num
import pytest
from lightguide import afk_filter, orafk_filter


@pytest.fixture
def data():
    import lightguide

    data = Path(lightguide.__file__).parent / "data" / "data-DAS-gfz2020wswf.npy"
    return num.load(data)


@pytest.fixture
def data_big():
    n = 2048
    return num.random.uniform(size=(n, n)).astype(num.float32)


def test_taper():
    window = afk_filter.triangular_taper(32, 4)
    assert window[32 // 2, 32 // 2] == 1.0
    assert window.shape == (32, 32)

    taper_rust = orafk_filter.triangular_taper((32, 32), (4, 4))
    num.testing.assert_almost_equal(window, taper_rust)


def test_plot_taper(show_plot):

    taper_rust = orafk_filter.triangular_taper((32, 64), (4, 10))

    if show_plot:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(taper_rust)
        plt.show()


@pytest.mark.skip
def test_benchmark_goldstein(benchmark, data_big):
    benchmark(afk_filter.afk_filter, data_big, 32, 14, 0.5)


@pytest.mark.skip
def test_benchmark_goldstein_rust(benchmark, data_big):
    benchmark(orafk_filter.afk_filter, data_big, 32, 14, 0.5, False)


def test_goldstein_rust(data):
    filtered_data_rust = orafk_filter.afk_filter(
        data, 32, 14, exponent=0.0, normalize_power=False
    )

    filtered_data_rust_rect = orafk_filter.afk_filter_rectangular(
        data, (32, 32), (14, 14), exponent=0.0, normalize_power=False
    )
    num.testing.assert_almost_equal(filtered_data_rust_rect, filtered_data_rust)
    num.testing.assert_allclose(data, filtered_data_rust, rtol=1.0)

    filtered_data_rust_rect = orafk_filter.afk_filter_rectangular(
        data, (32, 16), (14, 7), exponent=0.0, normalize_power=False
    )

    filtered_data_rust_rect = orafk_filter.afk_filter_rectangular(
        data, (32, 128), (14, 56), exponent=0.0, normalize_power=False
    )

    filtered_data_rust_rect = orafk_filter.afk_filter_rectangular(
        data, (32, 200), (14, 80), exponent=0.0, normalize_power=False
    )
