import matplotlib.pyplot as plt
import numpy as np


def plot_show():
    """
    Shows the current plot
    """

    plt.show()


def plot_connectivity_matrix(CIJ, n, plot_figure=None):
    """
    Plots a scatter matrix
    """

    plot = plt.figure(plot_figure).add_subplot(1, 1, 1)
    x, y = np.where(CIJ == 1)
    plot.axis([0, n, 0, n])
    plot.scatter(x, y, vmin=0, vmax=n)

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


def plot_firings(neuron_network, duration, plot_figure=None):
    """
    Plots the firing events of every neuron in each layer of the network
    """

    plt.figure(plot_figure)

    for index, layer in neuron_network.layers.items():
        plt.subplot(len(neuron_network.layers), 1, 1 + index)
        plt.scatter(layer.firings[:, 0], layer.firings[:, 1] + 1, marker='.')
        plt.xlim(0, duration)
        plt.ylabel('Neuron number')
        plt.ylim(0, layer.N + 1)
        plt.title('Population ' + str(index + 1) + ' firings')

    return plt
