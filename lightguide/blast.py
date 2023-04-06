from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    TypeVar,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, dates
from matplotlib.colors import Colormap
from pyrocko import io
from pyrocko.trace import Trace
from scipy import signal

from lightguide.utils import PathStr

from .filters import afk_filter
from .signal import decimation_coefficients

if TYPE_CHECKING:
    from matplotlib import image


logger = logging.getLogger(__name__)


MeasurementUnit = Literal[
    "strain rate",
    "strain",
    "acceleration",
    "velocity",
    "displacement",
    "velocity",
]


class Blast:
    data: np.ndarray
    start_time: datetime
    sampling_rate: float
    processing_flow: list

    unit: MeasurementUnit

    start_channel: int
    channel_spacing: float

    def __init__(
        self,
        data: np.ndarray,
        start_time: datetime,
        sampling_rate: float,
        start_channel: int = 0,
        channel_spacing: float = 0.0,
        unit: MeasurementUnit = "strain rate",
    ):
        self.data = data
        self.start_time = start_time
        self.sampling_rate = sampling_rate

        self.start_channel = start_channel
        self.channel_spacing = channel_spacing

        self.processing_flow = []

    @property
    def delta_t(self) -> float:
        return 1.0 / self.sampling_rate

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.n_samples * self.delta_t)

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def end_channel(self) -> int:
        return self.start_channel + self.n_channels

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples * self.delta_t

    def detrend(self, type: Literal["linear", "constant"] = "linear") -> None:
        """Demean and detrend the blast in-place.

        Args:
            type (Literal["linear", "constant"], optional): he type of detrending.
            If type == 'linear' (default), the result of a linear least-squares fit
            to data is subtracted from data. If type == 'constant',
            only the mean of data is subtracted.. Defaults to "linear".
        """
        self.data = signal.detrend(self.data, type=type, axis=1)

    def decimate(self, factor: int, demean: bool = True) -> None:
        """Decimate the trace by a factor, in-place.

        Args:
            factor (int): Decimation factor, must be > 1.
            demean (bool): Demean signal before decimation. Defaults to True
        """
        if factor == 1:
            return

        self.data = self.data.astype(np.float32, copy=False)
        if demean:
            self.data -= np.mean(self.data, axis=1, keepdims=True)

        if ((self.sampling_rate / factor) % 1.0) != 0:
            logger.warning(
                "Decimating to odd sampling rate %.2f", self.sampling_rate / factor
            )

        coeff_b, coeff_a, _ = decimation_coefficients(factor, filter_type="fir-remez")
        self.data = signal.lfilter(coeff_b, coeff_a, self.data, axis=1)[:, ::factor]
        self.sampling_rate /= factor

    def lowpass(
        self,
        cutoff_freq: float,
        order: int = 4,
        demean: bool = True,
        zero_phase: bool = False,
    ) -> None:
        """Lowpass filter the trace in-place using a Butterworth filter

        Args:
            cutoff_freq (float): cutoff frequency of the lowpass filter.
            order (int): order of the filter
            demean: (Optional) Mean is removed before filtering. Default is True.
            zero_phase (bool, optional): Use zero phase filter. Defaults to True.
        """
        sos = signal.butter(
            N=order,
            Wn=cutoff_freq,
            btype="lowpass",
            fs=self.sampling_rate,
            output="sos",
        )
        self.data = self.data.astype(np.float32, copy=False)
        if demean:
            self.data -= np.mean(self.data, axis=1, keepdims=True)
        if zero_phase:
            self.data = signal.sosfiltfilt(sos, self.data, axis=1)
        else:
            self.data = signal.sosfilt(sos, self.data, axis=1)

    def highpass(
        self,
        cutoff_freq: float,
        order: int = 4,
        demean: bool = True,
        zero_phase: bool = False,
    ) -> None:
        """Highpass filter the trace in-place using a Butterworth filter

        Args:
            cutoff_freq (float): cutoff frequency of the highpass filter.
            order (int): order of the filter. Default is 4.
            demean: (bool, optional) Remove mean before filtering. Default is True.
            zero_phase (bool, optional): Use zero phase filter. Defaults to True.
        """
        sos = signal.butter(
            N=order,
            Wn=cutoff_freq,
            btype="highpass",
            fs=self.sampling_rate,
            output="sos",
        )
        self.data = self.data.astype(np.float32, copy=False)
        if demean:
            self.data -= np.mean(self.data, axis=1, keepdims=True)

        if zero_phase:
            self.data = signal.sosfiltfilt(sos, self.data, axis=1)
        else:
            self.data = signal.sosfilt(sos, self.data, axis=1)

    def bandpass(
        self,
        min_freq: float,
        max_freq: float,
        order: int = 4,
        demean: bool = True,
        zero_phase: bool = False,
    ) -> None:
        """Apply band pass filter to the trace in-place

        Args:
            min_freq (float): Lower corner of the band pass filter in Hz.
            max_freq (float): Upper corner of the band pass filter in Hz.
            order (int, optional): Order of the filter. Defaults to 4.
            demean (bool, optional): Remove mean before filtering. Defaults to True.
            zero_phase (bool, optional): Use zero phase filter. Defaults to True.
        """
        sos = signal.butter(
            N=order,
            Wn=(min_freq, max_freq),
            btype="bandpass",
            fs=self.sampling_rate,
            output="sos",
        )
        self.data = self.data.astype(np.float32, copy=False)
        if demean:
            self.data -= np.mean(self.data, axis=1, keepdims=True)

        if zero_phase:
            self.data = signal.sosfiltfilt(sos, self.data, axis=1)
        else:
            self.data = signal.sosfilt(sos, self.data, axis=1)

    def afk_filter(
        self,
        window_size: int = 16,
        overlap: int = 7,
        exponent: float = 0.8,
        normalize_power: bool = False,
    ):
        """Adaptive frequency filter the blast in-place.

        The adaptive frequency filter (AFK) can be used to suppress incoherent noise
        in DAS data and other spatially coherent data sets.

        Args:
            window_size (int): Square window size. Defaults to 16.
            overlap (int): Overlap of neighboring windows. Defaults to 7.
            exponent (float): Filter strength, between 0.0 and 1.0. Defaults to 0.8.
            normalize_power (bool): Normalize the power. Defaults to True.
        """
        self.data = afk_filter(
            self.data.astype(np.float32),
            window_size=window_size,
            overlap=overlap,
            exponent=exponent,
            normalize_power=normalize_power,
        )

    def taper(self, alpha: float = 0.05) -> None:
        """Taper the trace in-place with a Tukey window, aka a tapered cosine window.

        Args:
            alpha (float, optional): Shape parameter of the Tukey window, representing
                the fraction of the window inside the cosine tapered region.
                If zero, the Tukey window is equivalent to a rectangular window.
                If one, the Tukey window is equivalent to a Hann window.
                Defaults to 0.05.
        """
        self.data = self.data.astype(np.float32, copy=False)
        window = signal.windows.tukey(self.data.shape[1], alpha=alpha)
        self.data *= window[np.newaxis, :]

    def one_bit_normalization(self) -> None:
        """Apply one-bit normalization on the trace."""
        self.data = np.sign(self.data)

    def mute_median(self, level: float = 3.0) -> None:
        """Mute signals in the data above a threshold.

        Args:
            level (float, optional): Median level to mute. Defaults to 3.0.
        """
        envelope = np.abs(signal.hilbert(self.data, axis=1))
        levels = np.mean(envelope, axis=1)
        cutoff_level = level * np.median(levels)
        self.data[envelope > cutoff_level] = 0.0

    def trim_channels(self, begin: int = 0, end: int = -1) -> None:
        """Trim the `Blast` to channels.

        Args:
            begin (int, optional): Begin channel. Defaults to 0.
            end (int, optional): End channel. Defaults to -1.
        """
        self.start_channel += begin
        self.data = self.data[begin:end]

    def trim_time(self, begin: float = 0.0, end: float = -1.0) -> None:
        if end < begin and end != -1.0:
            raise ValueError("Begin sample has to be before end")
        start_sample = int(begin // self.delta_t)
        end_sample = max(int(end // self.delta_t), -1)

        self.data = self.data[:, start_sample:end_sample]
        self.start_time += timedelta(seconds=begin)

    def to_strain(self, detrend: bool = True) -> Blast:
        """Convert the traces to strain.

        Args:
            detrend (bool, optional): Detrend trace before integration.
                Defaults to True.

        Raises:
            TypeError: Raised when the input blast is not in strain rate.

        Returns:
            Blast: As strain strain.
        """
        if self.unit == "strain":
            return self.copy()
        if self.unit != "strain rate":
            raise TypeError(f"Blast has a bad unit {self.unit}, expected 'strain_rate'")
        blast = self.copy()
        if detrend:
            blast.detrend()
        blast.data = np.cumsum(blast.data, axis=1)
        blast.unit = "strain"
        return blast

    def to_relative_velocity(self, detrend: bool = True) -> Blast:
        """Convert the traces to relative velocity.

        Args:
            detrend (bool, optional): Detrend trace before spatial differentiation.
                Defaults to True.

        Raises:
            TypeError: Raised when the input blast is not in strain rate.

        Returns:
            Blast: As strain strain.
        """
        blast = self.to_strain(detrend=detrend)
        if detrend:
            blast.detrend()
        blast.data = np.diff(blast.data, n=1, axis=0) / blast.channel_spacing
        blast.unit = "velocity"
        return blast

    def to_relative_displacement(self, detrend: bool = True) -> Blast:
        """Convert the traces to relative displacement.

        Args:
            detrend (bool, optional): Detrend trace before spatial integration.
                Defaults to True.

        Raises:
            TypeError: Raised when the input blast is not in strain rate.

        Returns:
            Blast: As strain strain.
        """
        blast = self.to_strain()
        if detrend:
            blast.detrend()
        blast.data = np.cumsum(blast.data, axis=0) * blast.channel_spacing
        blast.unit = "displacement"
        return blast

    def plot(
        self,
        axes: plt.Axes | None = None,
        cmap: str | Colormap = "seismic",
        show_date: bool = False,
        show_channel: bool = False,
    ) -> image.AxesImage:
        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes

        d_channel = self.channel_spacing
        if not d_channel:
            show_channel = True
        extent = (
            self.start_channel if show_channel else self.start_channel * d_channel,
            self.end_channel if show_channel else self.end_channel * d_channel,
            dates.date2num(self.end_time)
            if show_date
            else self.n_samples * self.delta_t,
            dates.date2num(self.start_time) if show_date else 0.0,
        )

        img = ax.imshow(
            self.data.T,
            aspect="auto",
            # interpolation="nearest",
            cmap=cmap,
            extent=extent,
            norm=colors.CenteredNorm(),
        )

        ax.set_ylabel("Time [s]")
        if show_date:
            ax.set_ylabel("Time UTC")
            ax.yaxis_date()

        ax.set_xlabel("Distance [m]")
        if show_channel:
            ax.set_xlabel("Channel #")

        if axes is None:
            plt.show()
        return img

    def copy(self) -> Blast:
        """Return a copy of the blast.

        Returns:
            Blast: Copied blast.
        """
        return deepcopy(self)

    def save_mseed(self, filename: PathStr) -> None:
        """Save as miniSeed.

        Args:
            filename (PathStr):
        """
        filename = Path(filename)
        traces = self.as_traces()
        io.save(traces, filename_template=str(filename), format="mseed")

    def as_traces(self) -> list[Trace]:
        """Converts the data in the Blast object into Pyrocko's Trace format.

        Returns:
            list[Trace]: A list of Pyrocko Trace instances,
                one for each channel in the data.
        """

        traces = []
        for icha in range(self.n_channels):
            channel = icha + self.start_channel
            traces.append(
                Trace(
                    ydata=self.data[icha],
                    tmin=self.start_time.timestamp(),
                    deltat=self.delta_t,
                    station=f"{channel:05d}",
                )
            )
        return traces

    @classmethod
    def from_pyrocko(cls, traces: list[Trace], channel_spacing: float = 0.0) -> Blast:
        """Create Blast from a list of Pyrocko traces.

        Args:
            traces (list[Trace]): List of input traces
            channel_spacing (float, optional): Spatial channel spacing in meter.
                Defaults to 0.0.

        Raises:
            ValueError: If input is odd.

        Returns:
            Blast: Assembled Blast.
        """
        if not traces:
            raise ValueError("Empty list of traces")

        traces = sorted(traces, key=lambda tr: int(tr.station))
        ntraces = len(traces)

        tmin = set()
        dtype = set()
        delta_t = set()
        nsamples = set()

        for tr in traces:
            tmin.add(tr.tmin)
            delta_t.add(tr.deltat)
            dtype.add(tr.ydata.dtype)
            nsamples.add(tr.ydata.size)

        if len(delta_t) != 1:
            raise ValueError(f"Sampling rate differs {delta_t}")
        if len(nsamples) != 1:
            raise ValueError(f"Traces number of samples differ {nsamples}")
        if len(dtype) != 1:
            raise ValueError(f"dtypes of traces differ {dtype}")
        if len(tmin) != 1 and np.abs(np.diff(list(tmin))).max() > 1e4:
            raise ValueError(f"Trace tmin differ {tmin}")

        data = np.zeros(shape=(ntraces, nsamples.pop()), dtype=dtype.pop())
        for itr, tr in enumerate(traces):
            data[itr, :] = tr.ydata

        return cls(
            data=data,
            start_time=datetime.fromtimestamp(tmin.pop(), tz=timezone.utc),
            sampling_rate=int(1.0 / delta_t.pop()),
            start_channel=min(int(tr.station) for tr in traces),
            channel_spacing=channel_spacing,
        )


TFun = TypeVar("TFun", bound=Callable[..., Any])


def shared_function(func: TFun) -> TFun:
    @wraps(func)
    def wrapper(self: Pack, *args, **kwargs) -> Any:
        for blast in self.blasts:
            func(blast, *args, **kwargs)

    wrapper.__doc__ = wrapper.__doc__.replace("Blast", "Pack")
    return cast(TFun, wrapper)


class Pack:
    blasts: set[Blast]

    def __init__(self, blasts: Iterable[Blast] | None = None) -> None:
        self.blasts = set(*blasts) if blasts else set()

    @property
    def n_blasts(self) -> int:
        return len(self.blasts)

    def add(self, blast: Blast) -> None:
        self.blasts.add(blast)

    def extend(self, blasts: Iterable[Blast]) -> None:
        for blast in blasts:
            self.add(blast)

    @property
    def start_time(self) -> datetime:
        return min(blast.start_time for blast in self.blasts)

    @property
    def end_time(self) -> datetime:
        return max(blast.end_time for blast in self.blasts)

    def copy(self) -> Pack:
        return deepcopy(self)

    def __getitem__(self, key) -> Blast:
        return list(self.blasts)[key]

    def __iter__(self) -> Iterator[Blast]:
        return iter(self.blasts)

    def __contains__(self, blast: Blast) -> bool:
        return blast in self.blasts

    def __len__(self) -> int:
        return self.n_blasts

    taper = shared_function(Blast.taper)
    detrend = shared_function(Blast.detrend)
    lowpass = shared_function(Blast.lowpass)
    highpass = shared_function(Blast.highpass)
    bandpass = shared_function(Blast.bandpass)
    afk_filter = shared_function(Blast.afk_filter)
    decimate = shared_function(Blast.decimate)

    trim_time = shared_function(Blast.trim_time)
    trim_channels = shared_function(Blast.trim_channels)

    mute_median = shared_function(Blast.mute_median)
    one_bit_normalization = shared_function(Blast.one_bit_normalization)
    afk_filter = shared_function(Blast.afk_filter)
    decimate = shared_function(Blast.decimate)

    trim_time = shared_function(Blast.trim_time)
    trim_channels = shared_function(Blast.trim_channels)

    to_strain = shared_function(Blast.to_strain)
    to_relative_velocity = shared_function(Blast.to_relative_velocity)
    to_relative_displacement = shared_function(Blast.to_relative_displacement)
