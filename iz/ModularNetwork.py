import numpy as np
import matplotlib.pyplot as plt


class ModularNetwork(object):

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

    def plot(self):
        """
        Uses pyplot to draw a plot of the connectivity matrix
        """

        plot = plt.figure().add_subplot(1, 1, 1)
        x, y = np.where(self.CIJ == 1)
        plot.axis([0, self.n, 0, self.n])
        plot.scatter(x, y, vmin=0, vmax=self.n)

        plt.show()
