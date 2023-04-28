import numpy as np
from scipy.signal import windows, hilbert
from pyrocko.trace import Trace


def get_peak_to_peak_amp(data) -> float:
    """Extracts peak to peak amplitude of input data, i.e. max and min of data
    Args:
        data (numpy.ndarray or list): wavelet to be investigated
    """
    data = data.copy().astype(float)
    window = windows.tukey(data.size, alpha=0.2)
    data *= window
    data -= np.mean(data)
    ampl = data.max() - data.min()
    return ampl


def get_spectral_amp(data) -> float:
    """Extracts spectral amplitude of input data, i.e. sum of fft spectrum.
    Args:
        data (numpy.ndarray or list): wavelet to be investigated
    """
    data = data.copy().astype(float)
    window = windows.tukey(data.size, alpha=0.2)
    data *= window
    data -= np.mean(data)
    spec = np.fft.rfft(data)
    spec /= spec.size
    power = np.sum(np.abs(spec) ** 2)
    return power


def get_frequency_spec(data, normalize: bool = False) -> float:
    """Calculates fft of given wavelet
    Args:
        data (numpy.ndarray or list): wavelet to be investigated
    """
    data = data.copy().astype(float)
    window = windows.tukey(data.size, alpha=0.2)
    data *= window
    data -= np.mean(data)
    spec = np.fft.rfft(data)
    values = np.abs(spec)
    if normalize:
        values /= values.max()
    # frequencies   = np.fft.rfftfreq(n=data.shape[-1],d=1./1000)
    return values


def get_envelope(data) -> float:
    """Calculates envelope of given wavelet
    Args:
        data (numpy.ndarray or list): wavelet to be investigated
    """
    data = data.copy().astype(float)
    data -= np.mean(data)
    window = windows.tukey(data.size, alpha=0.2)
    data *= window
    anal_sig = hilbert(data)
    env = np.abs(anal_sig)
    return np.sum(env)
