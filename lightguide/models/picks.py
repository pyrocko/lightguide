from pydantic import BaseModel
from typing import List, Tuple, Dict
from datetime import datetime
from pyrocko import marker
import numpy as np
from lightguide.utils import PathStr


class Picks(BaseModel):
    channel: list[int]
    time: list[datetime]
    correlation: list[float] = []
    kind: list[int] = []
    phase: list[str] = []

    def save_picks(self, filename: PathStr) -> None:
        """
        Saves picks as a pyrocko markerfile.

        Args:
            filename (str): path to filename
        """
        marker.save_markers(markers=self.as_markers(), filename=str(filename))

    def add(self, picks) -> "Picks":
        """Adds picks to existing picks object and returns new pick.
        Args:
            picks (Picks): picks to add
        """
        mlist = self.as_markers()
        mlist.extend(picks.as_markers())
        return Picks.from_pyrocko_markers(mlist)

    def as_markers(self) -> list[marker.PhaseMarker]:
        """
        Converts picks object to pyrocko markers.
        """
        channels = self.channel
        times = self.time
        kinds = self.kind
        phases = self.phase

        if not self.kind:
            kinds = [0] * len(channels)
        if not self.phase:
            phases = [""] * len(channels)

        markers = []
        for ch, ptime, kind, phase in zip(channels, times, kinds, phases):
            nslc_id = [("", "%05d" % ch, "", "")]
            tmin = ptime.timestamp()
            m = marker.PhaseMarker(
                nslc_ids=nslc_id, tmin=tmin, tmax=tmin, kind=kind, phasename=phase
            )
            markers.append(m)
        return markers

    def as_array(self):
        """
        Transforms Picks lists into numpy array.
        """
        N = len(self.channel)
        temp = []
        for _, values in self:
            if len(values) > 0:
                temp.append(values)
            else:
                temp.append(N * [None])
        return np.transpose(np.asarray(temp))

    # @classmethod
    def sort_by(self, attribute: str = "channel", reverse: bool = False):
        """
        Sort Picks by a given attribute, in place. Default: sorting by channel number.

        Args:
            attribute (str): attribute to be sorted by, default: channel (number). Options: 'channel', 'time','correlation','kind','phase'
            reverse (bool): sorting reversed or normal, default: False
        Returns:
            self: sorted by given criteria
        """
        # find column index of attribute to be sorted by
        for i, (attr, _) in enumerate(self):
            if attr == attribute:
                col = i

        # convert to array & check if attribute is specified
        temp = self.as_array()
        if temp[0, col] == None:
            print("Attribute not specified. Default: sorting by channel.")
            col = 0
        # do the sorting
        temp = temp[temp[:, col].argsort()]
        if reverse == True:
            temp = temp[::-1]

        # update entries of picks object
        for i, (attr, _) in enumerate(self):
            if temp[0, i] == None:
                setattr(self, attr, [])
            else:
                setattr(self, attr, list(temp[:, i]))

    def get_picktime(self, channel: int) -> datetime | None:
        """
        Returns time of pick of a selected channel (first occurence).

        Args:
            channel (int): name of channel
        Returns:
            pick time (datetime| None): pick time at channel (first occurence in list) of channel not in list, returns none
        """
        try:
            idx = self.channel.index(channel)
            return self.time[idx]
        except:
            return None

    @classmethod
    def from_pyrocko_markers(cls, markers: List) -> "Picks":
        """
        Converts list of pyrocko markers to a Picks object

        Agrs:
            markers (list): list of pyrocko markers
        """
        channels = ["%05d" % int(m.nslc_ids[0][1]) for m in markers]
        times = [m.tmin for m in markers]
        kinds = [m.kind for m in markers]
        phases = [m._phasename for m in markers]

        return Picks(channel=channels, time=times, kind=kinds, phase=phases)

    @classmethod
    def from_pyrocko_picks(cls, filename: PathStr) -> "Picks":
        """
        Loads pyrocko picks from file.

        Agrs:
            filename (str): filename to read
        """
        markers = marker.load_markers(filename)
        return Picks.from_pyrocko_markers(markers=markers)
