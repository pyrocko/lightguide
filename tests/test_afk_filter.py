import numpy as np
import pytest

from lightguide import filters


@pytest.fixture
def data_big():
    n = 2048
    rng = np.random.default_rng()
    return rng.uniform(size=(n, n)).astype(np.float32)


def test_taper():
    window = filters.triangular_taper_python(32, 4)
    assert window[32 // 2, 32 // 2] == 1.0
    assert window.shape == (32, 32)

    taper_rust = filters.triangular_taper((32, 32), (4, 4))
    np.testing.assert_almost_equal(window, taper_rust)


def test_plot_taper(show_plot):
    taper_rust = filters.triangular_taper((32, 64), (4, 10))

    if show_plot:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(taper_rust)
        plt.show()


@pytest.mark.skip
def test_benchmark_goldstein(benchmark, data_big):
    benchmark(filters.afk_filter_python, data_big, 32, 14, 0.5)


@pytest.mark.skip
def test_benchmark_goldstein_rust(benchmark, data_big):
    benchmark(filters.afk_filter, data_big, 32, 14, 0.5, False)


def test_goldstein_rust(data_eq: np.ndarray):
    filtered_data_rust = filters.afk_filter(
        data_eq, 32, 14, exponent=0.0, normalize_power=False
    )

    filtered_data_rust_rect = filters.afk_filter_rectangular(
        data_eq, (32, 32), (14, 14), exponent=0.0, normalize_power=False
    )
    np.testing.assert_almost_equal(filtered_data_rust_rect, filtered_data_rust)
    np.testing.assert_allclose(data_eq, filtered_data_rust, rtol=1.0)

    filtered_data_rust_rect = filters.afk_filter_rectangular(
        data_eq, (32, 16), (14, 7), exponent=0.0, normalize_power=False
    )

    filtered_data_rust_rect = filters.afk_filter_rectangular(
        data_eq, (32, 128), (14, 56), exponent=0.0, normalize_power=False
    )

    filtered_data_rust_rect = filters.afk_filter_rectangular(
        data_eq, (32, 200), (14, 80), exponent=0.0, normalize_power=False
    )
