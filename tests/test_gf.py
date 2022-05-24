from pathlib import Path

import numpy as np
import pytest
from lightguide import gf
from pyrocko import trace
from pyrocko.model import Location

km = 1e3


def test_fiber_model():

    fiber = gf.Fiber(
        lat=0.5,
        lon=0.5,
        store_id="test",
        coordinates=((0, 0), (100, 0)),
        channel_spacing=1.0,
    )
    assert fiber.distances.size == 2
    assert fiber.distances[0] == 0.0
    assert fiber.distances[1] == 100.0

    assert fiber.length == 100.0
    assert fiber.nchannels == 100

    northings, eastings, elevations, azis, dips, channels = fiber.interpolate_channels()
    assert channels.size == 100.0

    np.testing.assert_equal(azis, 0.0)
    np.testing.assert_equal(dips, 0.0)

    targets = fiber.get_targets()

    assert len(targets) == 100
    assert len(targets) == fiber.nchannels

    fiber = gf.Fiber(
        lat=0.0,
        lon=0.0,
        store_id="test",
        coordinates=(
            (0, 0, 0.0),
            (-20, 0, 20.0),
            (-40, 0, 40.0),
        ),
        channel_spacing=1.0,
        interpolation="multilinear",
    )

    northings, eastings, elevations, azis, dips, channels = fiber.interpolate_channels()
    np.testing.assert_equal(azis, 180.0)
    np.testing.assert_equal(dips, -45.0)

    fiber = gf.Fiber(
        lat=0.0,
        lon=0.0,
        store_id="test",
        coordinates=(
            Location(north_shift=0.0, east_shift=0, elevation=0),
            Location(north_shift=-100.0, east_shift=0, elevation=100),
        ),
        channel_spacing=1.0,
        interpolation="multilinear",
    )

    northings, eastings, elevations, azis, dips, channels = fiber.interpolate_channels()
    np.testing.assert_equal(azis, 180.0)
    np.testing.assert_equal(dips, -45.0)


def test_model_fiber(show_plot):
    import pyrocko.gf as pgf

    fiber = gf.Fiber(
        lat=0.0,
        lon=0.0,
        quantity="strain_rate",
        store_id="das_test",
        coordinates=((1100, 0), (2000, 1000), (2000, 3000)),
        channel_spacing=4.0,
        smoothing_sigma=30.0,
    )
    engine = gf.LocalEngine(use_config=True)

    source = pgf.DCSource(lat=0.0, lon=0.0, depth=2 * km, strike=45.0, dip=30.0)

    traces = engine.process_fiber(source, fiber)
    if show_plot:
        trace.snuffle(traces)


def test_process_fiber_fft(show_plot):
    import pyrocko.gf as pgf

    engine = gf.LocalEngine(use_config=True)

    fiber = gf.Fiber(
        lat=0.0,
        lon=0.0,
        quantity="strain",
        store_id="das_test",
        coordinates=((1100, 0), (2000, 1000), (2000, 3000)),
        channel_spacing=4.0,
        smoothing_sigma=30.0,
        spectral_differences=True,
    )

    source = pgf.DCSource(lat=0.0, lon=0.0, depth=2 * km, strike=45.0, dip=30.0)

    traces_fft = engine.process_fiber(source, fiber)
    for tr in traces_fft:
        tr.set_location("SSP")

    fiber.spectral_differences = False
    traces_grad = engine.process_fiber(source, fiber)
    for tr in traces_grad:
        tr.set_location("SGR")

    fiber.quantity = "displacement"
    traces_u = engine.process_fiber(source, fiber)
    for tr in traces_u:
        tr.set_location("U")

    tr_u = traces_u[0].copy()
    tr_u.set_ydata((traces_u[1].ydata - traces_u[0].ydata) / fiber.channel_spacing)
    tr_u.set_location("U2")

    if show_plot:
        trace.snuffle(traces_grad + traces_fft + traces_u + [tr_u])


@pytest.mark.skipif(
    not Path("das-iceland-locations-stations.txt").exists(),
    reason="could not find station file",
)
def test_process_fiber_from_stations(show_plot):
    import pyrocko.gf as pgf
    from pyrocko import model, trace

    stations = model.load_stations("das-iceland-locations-stations.txt")
    fiber = gf.Fiber.from_stations(stations, channel_spacing=10.0, store_id="das_test")

    src = pgf.DCSource(
        lat=63.82725,
        lon=-22.67712,
        north_shift=0.0 * km,
        east_shift=0.0 * km,
        depth=3 * km,
    )

    engine = gf.LocalEngine(use_config=True)
    traces = engine.process_fiber(src, fiber)

    if show_plot:
        trace.snuffle(traces)
