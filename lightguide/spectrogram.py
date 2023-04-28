from scipy.signal import windows  # , hilbert, medfilt2d
import numpy as np
from pyrocko import marker
import matplotlib.pyplot as plt


def get_spectrogram(
    blast,
    frequency_range: tuple = None,
    markerfile: str = None,
    time_window: int = 1,
    plot: bool = True,
    cut_all_traces: bool = True,
):
    """
    Creates a spectrogram from numpy array dataset

    INPUTS:
    blast: blast object
    frequency_range (tuple): defining frequency range of interest
    markerfile (str): path to markerfile of blast
    time_window (int): length of time window [s] used after pick-time
    plot (bool): quick plot of data
    cut_all_traces (bool): see description in 'cut_traces_markerfile'

    OUTPUTS:
    spectra_filt: array of all spectra (can be visualised using plt.imshow or plt.color)
    freq_vector: vector of frequencies contained in the data
    """

    def get_spectral_values(data, normalize: bool = False) -> float:
        data = data.copy()
        data = data.astype(float)
        window = windows.tukey(data.size, alpha=0.2)
        data *= window
        data -= np.mean(data)
        spec = np.fft.rfft(data)
        values = np.abs(spec)
        if normalize:
            values /= values.max()
        return values

    def select_frequencies(spec_list, freq_vector, frequency_range=None):
        """
        selects desired frequencies from list of spectra and creates a frequency and channel grid for plotting with plt.pcolor
        """
        if frequency_range == None:
            frequency_range = (freq_vector.min(), freq_vector.max())
            print("No specific frequency band selected.")
            print(
                f"Using frequencies between {freq_vector.min()} and {freq_vector.max()} Hz."
            )

        all_channels = np.linspace(1, len(spec_list), len(spec_list))
        freq_range = np.logical_and(
            freq_vector >= frequency_range[0], freq_vector <= frequency_range[1]
        )

        spectra_filt = []
        for spec_line in spec_list:
            spectra_filt.append(spec_line[freq_range])

        freq_grid, channel_grid = np.meshgrid(freq_vector[freq_range], all_channels)
        return spectra_filt, freq_grid, channel_grid

    def cut_traces_markerfile(das_data, markerfile, time_window, cut_all_traces=True):
        """
        Cuts traces from wave-onsets-information of marker file.
        INPUTS:
        das_data: list of pyrocko traces
        markerfile: filename of marker file
        time_window: length of time window to cut after pick-time
        cut_all_traces == False : cut only traces that have a pick-time in the marker file
        cut_all_traces == True : for traces without pick time, pick time of neighbouring trace is used
        """
        try:
            m = marker.load_markers(markerfile)
        except:
            print(
                f"File {markerfile} does not exist or is not a markerfile! Traces are not cut!"
            )
            return das_data

        # sort ascending by channel name
        m.sort(key=lambda x: x.nslc_ids[0][1])
        ma = m[0]
        cut_data = []
        if cut_all_traces == True:
            for tr in das_data:
                sta = tr.station
                ma = next((i for i in m if int(i.nslc_ids[0][1]) == int(sta)), ma)
                trchop = tr.chop(tmin=ma.tmin, tmax=ma.tmin + time_window)
                cut_data.append(trchop)

        else:
            for tr in das_data:
                sta = tr.station
                ma = next((i for i in m if int(i.nslc_ids[0][1]) == int(sta)), None)
                if ma == None:
                    continue
                trchop = tr.chop(tmin=ma.tmin, tmax=ma.tmin + time_window)
                cut_data.append(trchop)

        shortest_trace = len(min(cut_data, key=lambda x: len(x.ydata)).ydata)
        longest_trace = len(max(cut_data, key=lambda x: len(x.ydata)).ydata)
        if longest_trace - shortest_trace > 20:
            print(
                f"Attention: traces have different lengths of {shortest_trace} & {longest_trace} samples."
            )
        return cut_data

    das_data = blast.as_traces()

    # cut traces along markerfile
    if markerfile != None:
        das_data = cut_traces_markerfile(
            das_data=das_data,
            markerfile=markerfile,
            cut_all_traces=cut_all_traces,
            time_window=time_window,
        )

    shortest_trace = len(min(das_data, key=lambda x: len(x.ydata)).ydata)
    longest_trace = len(max(das_data, key=lambda x: len(x.ydata)).ydata)
    if longest_trace - shortest_trace > 20:
        print(
            f"Attention: traces have different lengths of {shortest_trace} & {longest_trace} samples."
        )

    # create frequency vector
    freq_vector = np.fft.rfftfreq(n=len(das_data[0].ydata), d=das_data[0].deltat)

    # calculate spectra for traces
    all_spec_vals = []
    for tr in das_data:
        vals = get_spectral_values(data=tr.ydata)
        #################################################
        all_spec_vals.append(vals)

    # only use selected frequencies
    spectra_filt, freq_grid, channel_grid = select_frequencies(
        all_spec_vals,
        freq_vector,
        frequency_range=frequency_range,
    )

    # check if plot-option is choosen
    if plot == True:
        # freq_grid = np.log(freq_grid)
        fig, ax = plt.subplots()
        # ax.set_xscale('log')
        im = ax.imshow(spectra_filt, norm="log")
        ax.set_aspect("auto")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Channel")
        ax.set_xticklabels([])
    return spectra_filt, freq_grid, channel_grid
