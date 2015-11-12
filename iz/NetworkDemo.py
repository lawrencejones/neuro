#!/usr/bin/env python

import argparse
import numpy as np

from NeuronNetwork import NeuronNetwork, add_dirac_pulse
from QuadraticLayer import QuadraticLayer
from IzhikevichLayer import IzhikevichLayer
from Plotters import plot_show, plot_membrane_potentials, plot_firings

neuron_models = {"quadratic": QuadraticLayer, "izhikevich": IzhikevichLayer}

parser = argparse.ArgumentParser(description='Runs simulation of a two layer neuron network')

parser.add_argument('-m', '--model', required=True, choices=neuron_models.keys(),
                    help='neuron model')
parser.add_argument('-d', '--duration', required=True, type=int,
                    help='simulation duration')
parser.add_argument('-I', '--base-current', type=int, default=20,
                    help='base current for first layer')

parser.add_argument('layers', metavar='N', type=int, nargs='+',
                    help='number of neurons in layer')

args = parser.parse_args()

net = NeuronNetwork.create_feed_forward(neuron_models[args.model], args.layers, scaling_factor=50)
VS = [np.zeros([args.duration, layer_size]) for layer_size in args.layers]

for t in xrange(args.duration):

    net.layers[0].I = args.base_current * np.ones(net.layers[0].N)

    for layer_index in range(1, len(args.layers)):
        net.layers[layer_index].I = np.zeros(net.layers[layer_index].N)

    net.tick(t)

    for layer_index in range(len(args.layers)):
        VS[layer_index][t] = net.layers[layer_index].V

for layer_index, layer in net.layers.items():
    add_dirac_pulse(VS[layer_index], layer.firings)

plot_membrane_potentials(VS, args.duration, 1)
plot_firings(net, args.duration, 2)

plot_show()
