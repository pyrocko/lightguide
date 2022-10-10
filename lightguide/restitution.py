from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

from .utils import traces_to_numpy_and_meta

if TYPE_CHECKING:
    from pyrocko.trace import Trace


def rad_to_de(
    traces: list[Trace], m_per_rad: float = 11.6e-9, copy: bool = False
) -> None:

    out_traces = []
    for tr in traces:
        if "gauge_length" not in tr.meta:
            raise AttributeError("gauge_length not in metadata")
        if copy:
            tr = tr.copy()
        tr.ydata *= m_per_rad
        tr.ydata /= tr.meta["gauge_length"] * tr.deltat
        tr.ydata /= 8192
        tr.meta["unit"] = "strain rate (m/s/m)"
        out_traces.append(tr)


def strainrate_to_strain(
    traces: list[Trace], detrend: bool = True, copy: bool = False
) -> list[Trace]:
    out_traces = []
    data, _ = traces_to_numpy_and_meta(traces)

    if detrend:
        signal.detrend(data, type="linear", axis=1, overwrite_data=True)
    data = np.cumsum(data, axis=1)

    for itr, tr in enumerate(traces):
        if copy:
            tr = tr.copy()
        tr.set_ydata(data[itr])
        tr.meta["unit"] = "strain (m/m)"
        out_traces.append(tr)

    return out_traces


def strainrate_to_acceleration_static_slowness(
    traces: list[Trace], slowness: float, copy: bool = False
) -> list[Trace]:
    out_traces = []
    for tr in traces:
        if copy:
            tr = tr.copy()
        tr.set_ydata(tr.ydata / slowness)
        tr.meta["unit"] = "acceleration (m/s^2)"
        out_traces.append(tr)

    return out_traces


def strainrate_to_velocity_static_slowness(
    traces: list[Trace], slowness: float, copy: bool = False
) -> list[Trace]:
    out_traces = strainrate_to_strain(traces, copy)
    for tr in out_traces:
        tr.set_ydata(tr.ydata / slowness)
        tr.meta["unit"] = "velocity (m/s)"

    return out_traces


def strainrate_to_relative_displacement(
    traces: list[Trace], copy: bool = False
) -> list[Trace]:
    traces = strainrate_to_strain(traces, copy)
    data, meta = traces_to_numpy_and_meta(traces)

    data = np.cumsum(data, axis=0) * meta.spatial_resolution
    for itr, tr in enumerate(traces):
        tr.set_ydata(data[itr])
        tr.meta["unit"] = "displacement (m)"

    return traces


def strainrate_to_relative_velocity(
    traces: list[Trace], copy: bool = False
) -> list[Trace]:
    traces = strainrate_to_strain(traces, copy)
    data, meta = traces_to_numpy_and_meta(traces)

    data = np.diff(data, n=1, axis=0) / meta.spatial_resolution
    for itr, tr in enumerate(traces):
        tr.set_ydata(data[itr])
        tr.meta["unit"] = "velocity (m/s)"

    return traces
