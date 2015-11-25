import numpy as np

from iz.ModularSmallWorldNetwork import ModularSmallWorldNetwork
from iz.ModularFocalNetwork import ModularFocalNetwork
from iz.NeuronNetwork import NeuronNetwork
from iz.IzhikevichLayer import IzhikevichLayer

from iz.NetworkSimulator import simulate
from iz.Plotters import plot_connectivity_matrix

"""
The experiment requires two layers of neurons, one for excitatory neurons and another for
inhibitory.

We therefore require rules to govern interaction between each layer and the layer themselves. These
rules include conduction delay, scaling factors, connection weights and connectivity matrixes.

Recall that layer.Param[n][i,j] specifies the param on the connection from neuron j in layer n to
the neuron i in layer.

layer[0] -- Excitatory

layer[0].S[0] : small world network with 1000 random intracommunity connections as a seed
layer[0].S[1] : diffuse all-to-all

layer[1] -- Inhibitory

layer[1].S[0] : 4 focal connections from exc->inh
layer[1].S[1] : diffuse all-to-all

"""

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
                                              no_of_ex_to_ex_edges).rewire_network(.2).CIJ)

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

simulate(net, 200, bg_lam=0.1, bg_scale=15)
