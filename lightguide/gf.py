from collections.abc import Iterable

import numpy as np
import pyrocko.orthodrome as od
from pyrocko import gf
from pyrocko.guts import Bool, Float, List, Object, String, Timestamp, Tuple
from pyrocko.model import Location
from scipy import interpolate, ndimage, signal

r2d = 180.0 / np.pi
km = 1e3

META = {
    "measure_length": None,
    "start_distance": None,
    "stop_distance": None,
    "gauge_length": None,
    "spatial_resolution": None,
    "geo_lat": None,
    "geo_lon": None,
    "geo_elevation": None,
    "channel": None,
    "unit": None,
}


class QuantityType(gf.meta.QuantityType):
    choices = gf.meta.QuantityType.choices + ["strain", "strain_rate"]


class Fiber(Object):
    lat = Float.T(default=0.0)
    lon = Float.T(default=0.0)

    codes = Tuple.T(
        4,
        String.T(),
        default=("", "", "", "HSF"),
        help="network, station, location and channel codes to be set on "
        "the response trace. If station code is empty it will be filled"
        " by the channel number",
    )
    quantity = QuantityType.T(
        default="strain",
        help="Measurement quantity type. If not given, it is guessed from the "
        "channel code. For some common cases, derivatives of the stored "
        "quantities are supported by using finite difference "
        "approximations (e.g. displacement to velocity or acceleration). "
        "4th order central FD schemes are used.",
    )

    elevation = Float.T(default=0.0, help="station surface elevation in [m]")

    store_id = gf.meta.StringID.T(
        optional=True,
        help="ID of Green's function store to use for the computation. "
        "If not given, the processor may use a system default.",
    )

    sample_rate = Float.T(
        optional=True,
        help="sample rate to produce. "
        "If not given the GF store's default sample rate is used. "
        "GF store specific restrictions may apply.",
    )

    tmin = Timestamp.T(
        optional=True,
        help="time of first sample to request in [s]. "
        "If not given, it is determined from the Green's functions.",
    )
    tmax = Timestamp.T(
        optional=True,
        help="time of last sample to request in [s]. "
        "If not given, it is determined from the Green's functions.",
    )
    interpolation = gf.meta.InterpolationMethod.T(
        default="multilinear",
        help="Interpolation method between Green's functions. Supported are"
        " ``nearest_neighbor`` and ``multilinear``",
    )

    coordinates = List.T(
        help="coordinates of the cable as ``pyrocko.model.Location`` or a tuple"
        " of (north_shift, east_shift, [elevation])."
    )

    channel_spacing = Float.T(default=4.0, help="Channel spacing [m].")
    smoothing_sigma = Float.T(
        default=10.0, help="Standard deviation for Gaussian kernel in [m]"
    )

    spectral_differences = Bool.T(
        default=False, help="Use spectral derivation for estimation of linear strain."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._locations = []
        for coord in self.coordinates:
            if isinstance(coord, Location):
                loc = coord
            elif isinstance(coord, Iterable) and len(coord) in (2, 3):
                loc = Location(
                    lat=self.lat,
                    lon=self.lon,
                    north_shift=coord[0],
                    east_shift=coord[1],
                    elevation=self.elevation if len(coord) == 2 else coord[2],
                )
            else:
                raise AttributeError(
                    "coordinates have to be a list of pyrocko.model.Location"
                    " or tuples of (north_shift, east_shifts, [elevation]),"
                    " not %s" % type(coord)
                )

            self._locations.append(loc)
        assert len(self._locations) > 1, "Fiber needs more than 1 coordinates"

    @classmethod
    def from_stations(cls, stations, **kwargs):
        return cls(coordinates=stations, **kwargs)

    @property
    def distances(self):
        locations = self._locations
        nlocs = len(locations)

        dists = [0.0]
        for il in range(1, nlocs):
            dists.append(locations[il - 1].distance_3d_to(locations[il]))

        return np.array(dists)

    @property
    def length(self):
        return np.sum(self.distances)

    @property
    def nchannels(self):
        dists = self.length
        return int(dists // self.channel_spacing)

    def interpolate_channels(self, start_channel=0):
        dists = np.cumsum(self.distances)

        interp_lat = interpolate.interp1d(
            dists, tuple(loc.effective_lat for loc in self._locations)
        )
        interp_lon = interpolate.interp1d(
            dists, tuple(loc.effective_lon for loc in self._locations)
        )
        interp_elevation = interpolate.interp1d(
            dists, tuple(loc.elevation for loc in self._locations)
        )

        interp_distances = np.arange(0.0, dists.max(), step=self.channel_spacing)

        channels = np.arange(interp_distances.size) + start_channel
        nchannels = channels.size

        lats = interp_lat(interp_distances)
        lons = interp_lon(interp_distances)
        elevations = interp_elevation(interp_distances)

        azis = np.empty(nchannels)
        azis[:-1] = od.azimuth_numpy(
            lats[:-1],
            lons[:-1],
            lats[1:],
            lons[1:],
        )
        azis[-1] = azis[-2]

        ds = np.full(nchannels, self.channel_spacing)
        dips = -np.arctan2(np.gradient(elevations), ds) * r2d

        return lats, lons, elevations, azis, dips, channels

    def get_targets(self):
        lats, lons, elevations, azis, dips, channels = self.interpolate_channels()
        nchannels = self.nchannels

        quantity = self.quantity
        if self.quantity in ("strain", "strain_rate"):
            quantity = "displacement"

        targets = []
        for icha in range(nchannels):
            codes = list(self.codes)
            if not codes[1]:
                codes[1] = "%05d" % icha

            t = gf.Target(
                lat=lats[icha],
                lon=lons[icha],
                elevation=elevations[icha],
                azimuth=azis[icha],
                dip=dips[icha],
                north_shift=0.0,
                east_shift=0.0,
                sample_rate=self.sample_rate,
                codes=codes,
                store_id=self.store_id,
                quantity=quantity,
                interpolation=self.interpolation,
            )
            targets.append(t)
        return targets

    def get_locations(self):
        return self._locations


class LocalEngine(gf.LocalEngine):
    def process_fiber(self, source, fiber, **kwargs):
        if not isinstance(source, Iterable):
            raise ValueError("Currently only one source is supported!")

        targets = fiber.get_targets()
        resp = self.process(source, targets, **kwargs)

        traces = resp.pyrocko_traces()
        ntraces = len(traces)
        deltat = traces[0].deltat

        all_times = [(tr.tmin, tr.tmax) for tr in traces]
        all_tmin = np.min(all_times)
        all_tmax = np.max(all_times)

        meta = {
            "measure_length": fiber.length,
            "start_distance": 0.0,
            "stop_distance": fiber.length,
            "gauge_length": fiber.channel_spacing,
            "spatial_resolution": fiber.channel_spacing,
        }

        for icha, (tr, target) in enumerate(zip(traces, targets)):
            tr.extend(tmin=all_tmin, tmax=all_tmax, fillmethod="repeat")
            tr.meta = meta.copy()
            tr.meta["channel"] = icha
            tr.meta["geo_lat"] = target.lat
            tr.meta["geo_lon"] = target.lon
            tr.meta["geo_elevation"] = target.elevation

        nsamples = set([tr.ydata.size for tr in traces])
        assert len(nsamples) == 1
        nsamples = nsamples.pop()

        # Spatial smoothing
        trs_data = np.array([tr.ydata for tr in traces])
        if fiber.smoothing_sigma:
            trs_data = ndimage.gaussian_filter1d(
                trs_data, sigma=fiber.smoothing_sigma / fiber.channel_spacing, axis=0
            )

        if fiber.quantity in ("strain_rate", "strain"):
            assert (
                ntraces > 3
            ), "Too few channels in fiber for finite-difference derivation"

            if fiber.spectral_differences:
                kappa = np.fft.fftfreq(ntraces, fiber.channel_spacing)

                trs_spec = np.fft.fft(trs_data, axis=0)
                trs_spec *= 2 * np.pi * kappa[:, np.newaxis] * 1j
                trs_data = np.fft.ifft(trs_spec, axis=0).real
            else:
                trs_data = np.gradient(trs_data, fiber.channel_spacing, axis=0)

        if fiber.quantity == "strain_rate":
            trs_data = np.gradient(trs_data, deltat, axis=1)

            signal.detrend(trs_data, type="linear", axis=1, overwrite_data=True)
        # if fiber.quantity == 'strain':
        #     trs_data = np.cumsum(
        #         trs_data, axis=1)

        for itr, tr in enumerate(traces):
            tr.set_ydata(trs_data[itr])

        return traces
