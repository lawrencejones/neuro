import sys
import numpy as np

from neuro.ModularSmallWorldNetwork import ModularSmallWorldNetwork
from neuro.ModularFocalNetwork import ModularFocalNetwork
from neuro.NeuronNetwork import NeuronNetwork
from neuro.IzhikevichLayer import IzhikevichLayer

from neuro.NetworkSimulator import simulate
from neuro.Plotters import plot_show, \
    plot_membrane_potentials, \
    plot_firings, plot_module_mean_firing_rate, plot_connectivity_matrix

if len(sys.argv) < 2:
    print('Missing rewiring probability!')

duration = int(sys.argv[2]) if len(sys.argv) == 3 else 1000
rewiring_p = float(sys.argv[1])

no_of_modules = 8
no_of_excitatory = no_of_modules * 100
no_of_ex_to_ex_edges = no_of_modules * 1000
no_of_inhibitory = 200

excitatory_layer = IzhikevichLayer(no_of_excitatory, fire_type='regular')
inhibitory_layer = IzhikevichLayer(no_of_inhibitory, fire_type='fast')

net = NeuronNetwork([excitatory_layer, inhibitory_layer])

net.connect_layers([0, 0],  # Ex <- Ex
                   scaling_factor=17,
                   delay=np.random.random_integers(1, 20, (no_of_excitatory, no_of_excitatory)),
                   S=ModularSmallWorldNetwork(no_of_modules,
                                              no_of_excitatory,
                                              no_of_ex_to_ex_edges).rewire_network(rewiring_p).CIJ)

net.connect_layers([1, 0],  # In <- Ex
                   scaling_factor=50,
                   delay=np.ones((no_of_inhibitory, no_of_excitatory)),
                   S=ModularFocalNetwork(no_of_modules, (no_of_inhibitory, no_of_excitatory), 4).CIJ)

net.connect_layers([0, 1],  # Ex <- In
                   scaling_factor=2,
                   delay=np.ones((no_of_excitatory, no_of_inhibitory)),
                   S=-np.random.random_sample((no_of_excitatory, no_of_inhibitory)))

net.connect_layers([1, 1],  # In <- In
                   scaling_factor=1,
                   delay=np.ones((no_of_inhibitory, no_of_inhibitory)),
                   S=-np.random.random_sample((no_of_inhibitory, no_of_inhibitory)))
np.fill_diagonal(inhibitory_layer.S[1], 0)

membrane_potentials, net = simulate(net, duration, bg_lam=0.01, bg_scale=15)

plot_firings([net.layers[0]], duration, 2)
# plot_module_mean_firing_rate(net.layers[0], no_of_modules, resolution=[20, 50])

plot_show()

