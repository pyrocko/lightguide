from pydantic import BaseModel
from typing import List, Tuple, Dict
from datetime import datetime
from pyrocko import marker


class Picks(BaseModel):
    channel: list[int] = [-1]
    time: list[datetime] = None
    correlation: list[float] = []
    kind: list[int] = []

    def save_picks(self, filename):
        """
        Saves picks as a pyrocko markerfile.

        Args:
            filename (str): path to filename
            channels (list): list of channelnames
            times (list of datetimes): list of pick times in datetime-format
        """
        channels = self.channel
        times = self.time
        kinds = self.kind

        if not self.kind:
            kinds = [0] * len(channels)

        markers = []
        for ch, ptime, kind in zip(channels, times, kinds):
            nslc_id = [("DAS", "%05d" % ch, "", "")]
            tmin = ptime.timestamp()
            m = marker.Marker(nslc_ids=nslc_id, tmin=tmin, tmax=tmin, kind=kind)
            markers.append(m)
        marker.save_markers(markers=markers, filename=filename)

    @classmethod
    def from_pyrocko_picks(cls, filename):
        """
        Loads pyrocko picks from file.

        Agrs:
            filename (str): filename to read
        """
        markers = marker.load_markers(filename)
        channels = [m.nslc_ids[0][1] for m in markers]
        times = [m.tmin for m in markers]
        kinds = [m.kind for m in markers]

        return Picks(channel=channels, time=times, kind=kinds)

    # def plot ()
