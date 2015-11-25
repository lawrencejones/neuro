"""
Examples
========

ModularFocalNetwork(8, [1600, 800], 4).plot() => 8 modules, 4 connections to each neuron
"""

import numpy as np
from Plotters import plot_connectivity_matrix


def range_from_base(base, size):
    return xrange(base, base + size)


class ModularFocalNetwork(object):

    def __init__(self, C, dim, focal_width):
        """
        Generates connectivity matrix for a modular network with...

        C -- # communities/modules
        dim -- dimensions of matrix, [nodes_in_target_layer, nodes_in_input_layer]
        focal_width -- how connections per node in target layer

        Each community will have an even number of nodes, where each node has focal_width
        connections from randomly chosen nodes in the input layer.

        CIJ[i,j] represents the connection from node j in input layer to node i in this layer.
        """

        self.C = C
        self.dim = dim
        self.module_dim = [layer_size / C for layer_size in dim]
        self.focal_width = focal_width
        self.CIJ = np.zeros(dim)

        for i in range(C):
            self.init_module(i)

    def init_module(self, module_index):
        """
        Initialises the target module with connections from the input layer.
        """

        target_dim, input_dim = self.module_dim

        input_nodes = range_from_base(module_index * input_dim, input_dim)
        target_nodes = range_from_base(module_index * target_dim, target_dim)

        for i in target_nodes:
            nodes_to_connect = np.random.choice(input_nodes, self.focal_width, replace=False)
            self.CIJ[i, nodes_to_connect] = 1

    def plot(self):
        """
        Uses pyplot to draw a plot of the connectivity matrix
        """

        plot_connectivity_matrix(self.CIJ, self.dim).show()
