from pydantic import BaseModel
from typing import List, Tuple, Dict
from datetime import datetime
from pyrocko import marker

from lightguide.utils import PathStr


class Picks(BaseModel):
    channel: list[int]
    time: list[datetime]
    correlation: list[float] = []
    kind: list[int] = []

    def save_picks(self, filename: PathStr) -> None:
        """
        Saves picks as a pyrocko markerfile.

        Args:
            filename (str): path to filename
        """
        marker.save_markers(markers=self.as_markers(), filename=str(filename))

    def as_markers(self) -> list:
        """
        Converts picks object to pyrocko markers
        """
        channels = self.channel
        times = self.time
        kinds = self.kind

        if not self.kind:
            kinds = [0] * len(channels)

        markers = []
        for ch, ptime, kind in zip(channels, times, kinds):
            nslc_id = [("", "%05d" % ch, "", "")]
            tmin = ptime.timestamp()
            m = marker.Marker(nslc_ids=nslc_id, tmin=tmin, tmax=tmin, kind=kind)
            markers.append(m)
        return markers

    @classmethod
    def from_pyrocko_picks(cls, filename: PathStr) -> "Picks":
        """
        Loads pyrocko picks from file.

        Agrs:
            filename (str): filename to read
        """
        markers = marker.load_markers(filename)
        channels = ["%05d" % int(m.nslc_ids[0][1]) for m in markers]
        times = [m.tmin for m in markers]
        kinds = [m.kind for m in markers]

        return Picks(channel=channels, time=times, kind=kinds)
