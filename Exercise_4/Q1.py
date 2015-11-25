import sys
import numpy as np

from iz.ModularSmallWorldNetwork import ModularSmallWorldNetwork
from iz.ModularFocalNetwork import ModularFocalNetwork
from iz.NeuronNetwork import NeuronNetwork
from iz.IzhikevichLayer import IzhikevichLayer

from iz.NetworkSimulator import simulate
from iz.Plotters import plot_show, \
    plot_membrane_potentials, \
    plot_firings, plot_module_mean_firing_rate

if len(sys.argv) < 2:
    print('Missing rewiring probability!')

duration = int(sys.argv[2]) if len(sys.argv) == 3 else 1000
rewiring_p = float(sys.argv[1])

no_of_modules = 8
no_of_excitatory = no_of_modules * 100
no_of_ex_to_ex_edges = no_of_modules * 1000
no_of_inhibitory = no_of_modules * 200

excitatory_layer = IzhikevichLayer(no_of_excitatory, fire_type='regular')
inhibitory_layer = IzhikevichLayer(no_of_inhibitory, fire_type='fast')

net = NeuronNetwork([excitatory_layer, inhibitory_layer])

net.connect_layers([excitatory_layer, excitatory_layer],
                   scaling_factor=17,
                   delay=np.random.random_integers(
                       1, 20, (no_of_excitatory, no_of_excitatory)),
                   S=ModularSmallWorldNetwork(no_of_modules,
                                              no_of_excitatory,
                                              no_of_ex_to_ex_edges).rewire_network(rewiring_p).CIJ)

net.connect_layers([excitatory_layer, inhibitory_layer],
                   scaling_factor=50,
                   delay=1,
                   S=ModularFocalNetwork(no_of_modules, (no_of_inhibitory, no_of_excitatory), 4).CIJ)

net.connect_layers([inhibitory_layer, excitatory_layer],
                   scaling_factor=2,
                   delay=1,
                   S=-np.random.random_sample((no_of_excitatory, no_of_inhibitory)))

net.connect_layers([inhibitory_layer, inhibitory_layer],
                   scaling_factor=1,
                   delay=1,
                   S=-np.random.random_sample((no_of_inhibitory, no_of_inhibitory)))
np.fill_diagonal(inhibitory_layer.S[1], 0)

membrane_potentials, net = simulate(net, duration, base_current=3, bg_lam=0.1, bg_scale=15)

# plot_module_mean_firing_rate(net.layers[0], no_of_modules, resolution=[20, 50])
# plot_membrane_potentials(membrane_potentials, duration, 1)
plot_firings(net, duration, 2)

plot_show()

