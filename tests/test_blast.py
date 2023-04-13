from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pyrocko.trace import Trace

from lightguide.blast import Blast


def test_signaling(random_blast: Blast, show_plot: bool):
    torrent = random_blast
    assert torrent.delta_t
    assert torrent.n_channels < torrent.n_samples

    orig_blast = torrent.copy()

    def check_identity(new_blast: Blast, name: str):
        assert new_blast.n_channels == orig_blast.n_channels, f"bad shape {name}"
        assert new_blast.n_samples == orig_blast.n_samples, f"bad shape {name}"
        if show_plot:
            fig, axes = plt.subplots(2, 1)
            orig_blast.plot(axes=axes[0])
            new_blast.plot(axes=axes[1])
            fig.suptitle(name)
            plt.show()

    torrent = orig_blast.copy()
    torrent.detrend()
    check_identity(torrent, "Detrend")

    torrent = orig_blast.copy()
    torrent.lowpass(0.1)
    check_identity(torrent, "Lowpass")

    torrent = orig_blast.copy()
    torrent.highpass(2.0)
    check_identity(torrent, "highpass")

    torrent = orig_blast.copy()
    torrent.bandpass(1.0, 20.0)
    check_identity(torrent, "bandpass")

    torrent = orig_blast.copy()
    torrent.decimate(factor=1)
    check_identity(torrent, "decimate 1x")

    torrent = orig_blast.copy()
    torrent.decimate(factor=4)
    assert torrent.sampling_rate == orig_blast.sampling_rate / 4

    torrent = orig_blast.copy()
    torrent.taper(alpha=0.3)
    check_identity(torrent, "taper")


def test_from_pyrocko():
    rng = np.random.default_rng()
    trace = Trace(ydata=rng.integers(-1000, 1000, size=1000), station="123")
    Blast.from_pyrocko([trace] * 20)


def test_normalizations(random_blast: Blast):
    torrent = random_blast.copy()
    torrent.one_bit_normalization()

    torrent = random_blast.copy()
    torrent.mute_median()
