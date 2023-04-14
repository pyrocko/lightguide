from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import wraps
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
    """A Blast represents a time-space patch from a DAS recording."""

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
    ) -> None:
        """Create a new blast from NumPy array.

        Args:
            data (np.ndarray): Data as a 2D array.
            start_time (datetime): Start time of the blast.
            sampling_rate (float): Sampling rate in Hz.
            start_channel (int, optional): Start channel. Defaults to 0.
            channel_spacing (float, optional): Channel spacing in meters.
                Defaults to 0.0.
            unit (MeasurementUnit, optional): Measurement unit.
                Defaults to "strain rate".
        Raises:
            ValueError: When start_time has no time zone information.
        """
        if not start_time.tzinfo:
            raise ValueError("start_time has no time zone information.")

        self.data = data
        self.start_time = start_time
        self.sampling_rate = sampling_rate
        self.unit = unit

        self.start_channel = start_channel
        self.channel_spacing = channel_spacing

        self.processing_flow = []

    @property
    def delta_t(self) -> float:
        """Sampling interval in seconds."""
        return 1.0 / self.sampling_rate

    @property
    def end_time(self) -> datetime:
        """End time of the Blast."""
        return self.start_time + timedelta(seconds=self.n_samples * self.delta_t)

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.data.shape[0]

    @property
    def end_channel(self) -> int:
        """End Channel."""
        return self.start_channel + self.n_channels

    @property
    def n_samples(self) -> int:
        """Number of Samples."""
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.n_samples * self.delta_t

    def get_trace(self, channel: int) -> np.ndarray:
        """Get data from a singular channel.

        Args:
            channel (int): Channel number.

        Returns:
            np.ndarray: 1D Trace.
        """
        if not self.start_channel <= channel < self.end_channel:
            raise ValueError(f"Channel {channel} is out of bounds")
        return self.data[channel - self.start_channel]

    def _time_to_sample(self, time: datetime) -> int:
        """Get sample index for a time.

        Args:
            time (datetime): Time of desired sample

        Raises:
            ValueError: Raised when time is out of range.

        Returns:
            int: Sample index
        """
        if not self.start_time <= time <= self.end_time:
            raise ValueError("Time is out of range.")
        dt = time - self.start_time
        return int(dt.total_seconds() // self.delta_t)

    def _sample_to_time(self, index: int) -> datetime:
        """Get sample index for a time.

        Args:
            time (datetime): Time of desired sample

        Raises:
            ValueError: Raised when time is out of range.

        Returns:
            int: Sample index
        """
        dt = timedelta(seconds=index * self.delta_t)
        return self.start_time + dt

    def detrend(self, type: Literal["linear", "constant"] = "linear") -> None:
        """Demean and detrend in time-domain in-place.

        Args:
            type (Literal["linear", "constant"], optional): he type of detrending.
            If type == 'linear' (default), the result of a linear least-squares fit
            to data is subtracted from data. If type == 'constant',
            only the mean of data is subtracted.. Defaults to "linear".
        """
        self.data = signal.detrend(self.data, type=type, axis=1)

    def decimate(self, factor: int, demean: bool = True) -> None:
        """Decimate in time-domain by a factor, in-place.

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
        """Apply low-pass filter in-place using a Butterworth filter.

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
        """Apply high-pass filter in-place using a Butterworth filter.

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
        """Apply band-pass filter in-place using a Butterworth filter.

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
        exponent: float = 0.8,
        window_size: int = 16,
        overlap: int = 7,
        normalize_power: bool = False,
    ) -> None:
        """Apply adaptive frequency filter (AFK) in-place.

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

    def follow_phase(
        self,
        pick_time: datetime,
        pick_channel: int,
        window_size: int | tuple[int, int] = 50,
        threshold: float = 5e-1,
        max_shift: int = 20,
    ) -> tuple[np.ndarray, list[datetime], np.ndarray]:
        """Follow a phase pick through a Blast.

        This function leverages the spatial coherency of a DAS data set. Following a
        phase through iterative cross-correlation of neighboring traces:

        1. Get windowed root pick template.
        2. Calculate normalized cross correlate with downwards neighbor.
        3. Evaluate maximum x-correlation in allowed window (max_shift).
        4. Update template trace and go to 2.

        5. Repeat for upward neighbors.

        Args:
            pick_time (datetime): Initial pick in the data set.
            pick_channel (int): Corresponding channel for the initial pick.
            window_size (int | tuple[int, int], optional): Window size around the pick
                for the correlation template. If an integer is given it is the
                half-width. A Tuple gives pre- and post-windows. Defaults to 50.
            threshold (float, optional): Picking threshold for the cross correlation.
                Defaults to 5e-1.
            max_shift (int, optional): Maximum allowed shift in samples for
                neighboring picks. Defaults to 20.

        Returns:
            tuple[np.ndarray, list[datetime], np.ndarray]: Tuple of channel number,
                pick time and maximum correlation coefficient.
        """
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        pick_channel -= self.start_channel
        root_idx = self._time_to_sample(pick_time)

        # Ensure the window is odd-sized with the pick in center.
        root_template = self.get_trace(pick_channel)[
            root_idx - window_size[0] : root_idx + window_size[1] + 1
        ].copy()
        template_taper = signal.windows.tukey(root_template.size, alpha=0.1)

        pick_channels, pick_times, pick_correlations = [], [], []

        def prepare_template(data: np.ndarray) -> np.ndarray:
            return data * template_taper

        def correlate(data: np.ndarray, direction: Literal[1, -1] = 1) -> None:
            template = root_template.copy()
            index = root_idx

            for ichannel, trace in enumerate(data):
                template = prepare_template(template)
                norm = np.sqrt(np.sum(template**2)) * np.sqrt(np.sum(trace**2))
                correlation = np.correlate(trace, template, mode="same")
                correlation = np.abs(correlation / norm)

                # Take maximum only in allowed window
                phase_idx = (
                    np.argmax(correlation[index - max_shift : index + max_shift + 1])
                    + index
                    - max_shift
                )

                phase_correlation = correlation[phase_idx]
                phase_time = self._sample_to_time(int(phase_idx))

                if phase_correlation < threshold:
                    continue

                # Avoid the edges
                if not window_size[0] < phase_idx < self.n_samples - window_size[1]:
                    continue

                pick_times.append(phase_time)
                pick_channels.append(pick_channel + ichannel * direction)
                pick_correlations.append(phase_correlation)

                template = trace[
                    phase_idx - window_size[0] : phase_idx + window_size[1] + 1
                ].copy()
                index = phase_idx

        correlate(self.data[pick_channel:])
        correlate(self.data[: pick_channel - 1][::-1], direction=-1)

        pick_channels = np.array(pick_channels) + self.start_channel
        pick_correlations = np.array(pick_correlations)

        return pick_channels, pick_times, pick_correlations

    def taper(self, alpha: float = 0.05) -> None:
        """Taper in time-domain and in-place with a Tukey window.

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
        """Mute signals in the data above a threshold in-place.

        Args:
            level (float, optional): Median level to mute. Defaults to 3.0.
        """
        envelope = np.abs(signal.hilbert(self.data, axis=1))
        levels = np.mean(envelope, axis=1)
        cutoff_level = level * np.median(levels)
        self.data[envelope > cutoff_level] = 0.0

    def trim_channels(self, begin: int = 0, end: int = -1) -> Blast:
        """Trim the Blast to channels and return a copy.

        Args:
            begin (int, optional): Begin channel. Defaults to 0.
            end (int, optional): End channel. Defaults to -1.

        Returns:
            Blast: Trimmed Blast.
        """
        blast = self.copy()
        blast.start_channel += begin
        blast.data = blast.data[begin:end]
        return blast

    def trim_time(self, begin: float = 0.0, end: float = -1.0) -> Blast:
        """Trim channel to time frame and return a copy.

        Args:
            begin (float, optional): Begin time. Defaults to 0.0.
            end (float, optional): End time. Defaults to -1.0.

        Raises:
            ValueError: Raised when begin and end are ill behaved.

        Returns:
            Blast: Trimmed Blast.
        """
        if end < begin and end != -1.0:
            raise ValueError("Begin sample has to be before end")
        blast = self.copy()
        start_sample = int(begin // self.delta_t)
        end_sample = max(int(end // self.delta_t), -1)

        blast.data = blast.data[:, start_sample:end_sample]
        blast.start_time += timedelta(seconds=begin)
        return blast

    def to_strain(self, detrend: bool = True) -> Blast:
        """Convert the traces to strain.

        Args:
            detrend (bool, optional): Detrend trace before integration.
                Defaults to True.

        Raises:
            TypeError: Raised when the input Blast is not in strain rate.

        Returns:
            Blast: In strain strain.
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
            Blast: As relative velocity.
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
            TypeError: Raised when the input Blast is not in strain rate.

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
        normalize_traces: bool = True,
        cmap: str | Colormap = "seismic",
        show_date: bool = False,
        show_channel: bool = False,
    ) -> image.AxesImage:
        """Plot data of the blast.

        Args:
            axes (plt.Axes | None, optional): Optional axis to plot into. If not given,
                `plt.show()` is called. Defaults to None.
            normalize_traces (bool, optional): Normalize traces in time dimension.
                Defaults to True.
            cmap (str | Colormap, optional): Matplotlib colormap. Defaults to "seismic".
            show_date (bool, optional): Shot absolute dates in UTC. Defaults to False.
            show_channel (bool, optional): Show channels instead of meters.
                Defaults to False.

        Returns:
            image.AxesImage: Decorated axes.
        """
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

        data = self.data.copy()
        if normalize_traces:
            data /= np.abs(data.max(axis=1, keepdims=True))

        img = ax.imshow(
            data.T,
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
        """Save the Blast as miniSeed.

        Args:
            filename (PathStr): File to write to.
        """
        io.save(self.as_traces(), filename_template=str(filename), format="mseed")

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
    def from_pyrocko(cls, traces: list[Trace], channel_spacing: float = 4.0) -> Blast:
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

    @classmethod
    def from_miniseed(cls, file: PathStr, channel_spacing: float = 4.0) -> Blast:
        """Load Blast from miniSEED.

        Args:
            file (PathStr): miniSEED file to load from.
            channel_spacing (float, optional): Channel spacing in meters.
                Defaults to 4.0.

        Returns:
            Blast: Produced Blast.
        """
        from pyrocko import io

        traces = io.load(str(file), format="mseed")
        return cls.from_pyrocko(traces, channel_spacing=channel_spacing)


TFun = TypeVar("TFun", bound=Callable[..., Any])


def shared_function(func: TFun) -> TFun:
    @wraps(func)
    def wrapper(self: Pack, *args, **kwargs) -> Any:
        for blast in self.blasts:
            func(blast, *args, **kwargs)

    if isinstance(wrapper.__doc__, str):
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
