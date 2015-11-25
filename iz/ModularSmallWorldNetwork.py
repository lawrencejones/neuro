"""
Examples
========

ModularSmallWorldNetwork(120, 1200, 6).rewire_network(0.2).plot() => 6 modules, random rewiring
"""

import numpy as np
from Plotters import plot_connectivity_matrix


def select_random_int(low, high, not_allowed=None):
    """
    Generates an integer between low and high, that is not equal to not_allowed
    """

    r = np.random.randint(low, high)
    while r == not_allowed:
        r = np.random.randint(low, high)

    return r


class ModularSmallWorldNetwork(object):

    def __init__(self, n, m, C):
        """
        Generates a connectivity matrix for a modular network with the following properties...

        n -- # nodes in the network
        m -- # edges in network
        C -- # communities/modules

        Each community will host n/C nodes, with m/C intra-community edges.
        """

        self.n = n
        self.m = m
        self.C = C

        self.CIJ = np.zeros([n, n])

        for i in range(C):
            self.init_module(i)

    def init_module(self, module_index, no_of_edges=None):
        """
        Initializes module with no_of_edges randomly distributed one-way edges.
        """

        no_of_edges = self.m / self.C if no_of_edges is None else no_of_edges
        nodes_in_module = self.n / self.C

        lower = module_index * nodes_in_module
        upper = lower + nodes_in_module

        count = 0

        while count < no_of_edges:
            i, j = np.random.randint(lower, upper, (2))
            if (i != j) and self.CIJ[i][j] == 0:
                count = count + 1
                self.CIJ[i][j] = 1

    def rewire_network(self, p):
        """
        Rewires the network edges with probability p to a node in a different module
        """

        nodes_in_module = self.n / self.C

        for i, j in self.connected_neurons():
            if np.random.random_sample() < p:
                target_module = select_random_int(0, self.C, not_allowed=i % nodes_in_module)
                target_node = target_module * nodes_in_module + np.random.randint(nodes_in_module)

                self.CIJ[i][j] = 0
                self.CIJ[i][target_node] = 1

        return self

    def connected_neurons(self):
        """
        Returns iterable tuples of (i, j), where there is a connection from neuron i to neuron j in
        the connectivity matrix
        """

        return zip(*np.where(self.CIJ == 1))

    def plot(self):
        """
        Uses pyplot to draw a plot of the connectivity matrix
        """

        plot_connectivity_matrix(self.CIJ, self.n).show()
