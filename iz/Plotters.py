import matplotlib.pyplot as plt
import numpy as np


def plot_show():
    """
    Shows the current plot
    """

    plt.show()


def plot_connectivity_matrix(CIJ, title="", plot_figure=None):
    """
    Plots a scatter matrix
    """

    plot = plt.figure(plot_figure).add_subplot(1, 1, 1)
    plt.title(title)
    x, y = np.where(CIJ == 1)
    plot.axis([0, len(CIJ), 0, len(CIJ[0])])
    plot.scatter(x, y)

    return plt


def plot_module_mean_firing_rate(layer, no_of_modules, resolution=None, title="", plot_figure=None):
    """
    Plots the mean firing

    no_of_modules -- # of modules to run mean firing rate for
    resolution -- [sample_every_n_steps, window_size_of_sample]
    """

    n_steps, window_size = resolution
    window_buffer = window_size / 2
    max_spike_time = np.max(layer.firings[:, 0])
    duration = 100 * (1 + max_spike_time / 100)
    firings = layer.firings

    sampling_ts = range(window_buffer, duration - window_buffer, n_steps)
    firing_rates = np.zeros((len(sampling_ts), no_of_modules))
    module_size = layer.N / no_of_modules

    for i, t in enumerate(sampling_ts):

        firings_after_start = firings[firings[:, 0] > t - window_buffer]
        firings_in_window = firings_after_start[firings_after_start[:, 0] < t + window_buffer]

        for module_index, module_base in enumerate(range(0, layer.N, module_size)):
            firings_from_module = np.where(np.logical_and(
                firings_in_window >= module_base,
                firings_in_window < module_base + module_size))[0]
            firing_rates[i][module_index] = len(firings_from_module)

    plt.figure(plot_figure).add_subplot(1, 1, 1)
    plt.title(title)
    plt.ylabel('Mean firing rate')
    plt.xlabel('Time (ms) + 0s')
    plt.plot(sampling_ts, firing_rates)

    return plt


def plot_membrane_potentials(population_vs, duration, plot_figure=None):
    """
    Plots the neuron membrane potentials by population
    """

    plt.figure(plot_figure)

    for index, V in enumerate(population_vs):
        plt.subplot(len(population_vs), 1, 1 + index)
        plt.plot(range(duration), V)
        plt.title('Population ' + str(index + 1) + ' membrane potentials')
        plt.ylabel('Voltage (mV)')
        plt.ylim([-90, 40])

    return plt


def plot_firings(layer, duration, title="", plot_figure=None):
    """
    Plots the firing events of every neuron in the given layer
    """

    plt.figure(plot_figure)

    plt.subplot(1, 1, 1)
    plt.scatter(layer.firings[:, 0], layer.firings[:, 1] + 1, marker='.')
    plt.xlim(0, duration)
    plt.title(title)
    plt.ylabel('Neuron number')
    plt.xlabel('Time (ms) + 0s')
    plt.ylim(0, layer.N + 1)

    return plt
