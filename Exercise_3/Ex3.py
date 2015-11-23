from NetworkWattsStrogatz import NetworkWattsStrogatz
from PlotConnectivity import PlotConnectivity

import bct
import sys
import numpy as np


def global_eff(CIJ, bidirectional=False):
    return 1.0 / (len(CIJ) * (len(CIJ) - 1)) * np.sum(1.0 / min_hop_count(CIJ))


def local_eff(CIJ):
    return [global_eff(neighbourhood)
            for neighbourhood in map(lambda n: network_of_neighbours(CIJ, n), range(len(CIJ)))]


def neighbours_of(CIJ, i):
    return np.where(CIJ[i] > 0)[0]


def min_hop_count(CIJ):
    mhc = bct.distance_bin(CIJ)
    mhc[mhc == 0] = np.Inf

    return mhc


def network_of_neighbours(CIJ, i):
    neighbours = np.where(CIJ[i] > 0)[0]
    return CIJ[neighbours][:, neighbours]


N = int(sys.argv[1])
k = int(sys.argv[2])
p = float(sys.argv[3])

net = NetworkWattsStrogatz(N, k, p)

print("BrainConnectivityToolbox...\n")
print("\tEff[Global] = " + str(bct.efficiency_bin(net)))
print("\tEff[Local]  = " + str(bct.efficiency_bin(net, local=True)))

print("\n\nHomebrew...\n")
print("\tEff[Global] = " + str(global_eff(net)))
print("\tEff[Local]  = " + str(local_eff(net)))

PlotConnectivity(net)

