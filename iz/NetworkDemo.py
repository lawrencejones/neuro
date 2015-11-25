#!/usr/bin/env python

import argparse

from QuadraticLayer import QuadraticLayer
from IzhikevichLayer import IzhikevichLayer

from NetworkSimulator import simulate
from NeuronNetwork import NeuronNetwork
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
membrane_potentials, net = simulate(net, args.duration, base_current=args.base_current)

plot_membrane_potentials(membrane_potentials, args.duration, 1)
plot_firings(net, args.duration, 2)

plot_show()
