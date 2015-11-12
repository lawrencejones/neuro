#!/usr/bin/env python

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

from Environment import Environment
from NeuronNetwork import NeuronNetwork
from IzhikevichLayer import IzhikevichLayer

motor_max = 40
wheel_velocity_min = 0.025
wheel_velocity_max = wheel_velocity_min + wheel_velocity_min / 6.0

conduction_delay = 4
max_conduction_delay = 5


def hetrogenize_layer(layer):

    """
    Initializes an Izhikevich layer with parameters that are randomised appropriately for a
    Braitenburg network.
    """

    seed = rn.rand(layer.N)

    layer.C = layer.C + 15 * (seed ** 2)
    layer.D = layer.D - 6 * (seed ** 2)

    return layer


class Vehicle(object):

    def __init__(self, no_of_sensory_neurons=4, no_of_motor_neurons=4):

        self.net = self._construct_network([no_of_sensory_neurons, no_of_motor_neurons])
        self.x, self.y = 0, 0
        self.w = np.pi / 4  # 45 degrees

    def move(self, env, ul, ur):
        """
        Given an environment and both wheel velocities, will compute the new position of the vehicle
        inside the environment and update the vehicles internal position.
        """

        axel_length = 1

        bearing_left, bearing_right = (ul * wheel_velocity_max), (ur * wheel_velocity_max)
        bearing = (bearing_left + bearing_right) / 2.0
        c = bearing_right - bearing_left

        dx = bearing * np.cos(self.w)
        dy = bearing * np.sin(self.w)
        dw = np.arctan2(c, axel_length)

        x = np.mod(self.x + dt * dx, env.x_max)
        y = np.mod(self.y + dt * dy, env.y_max)
        w = self.w + dt * dw

        w = np.mod(w + np.pi, 2 * np.pi) - np.pi
        w = (2 * np.pi + w) if w < 0 else w

        self.x, self.y, self.w = x, y, w

        return self

    def _construct_network(self, layer_dimensions):
        """
        Creates a network of four Izhikevich layers, to represent the four layers that control a
        Braitenburg vehicle.

        Layers 0 and 1 represent the sensory input layers, while 2 and 3 are output layers connected
        to motors.
        """

        scaling_factor = 50 / np.sqrt(layer_dimensions[0])

        # Initialize [LeftSensory, RightSensory, LeftMotor, RightMotor]
        net = NeuronNetwork([hetrogenize_layer(IzhikevichLayer(layer_dimensions[i]))
                            for i in [0, 0, 1, 1]],
                            d_max=max_conduction_delay)

        # Connect sensory networks to alternate motor networks, in order to have attractive movement
        for sensory, motor in [[0, 3], [1, 2]]:
            net.connect_layers([sensory, motor],
                               scaling_factor=scaling_factor,
                               delay=conduction_delay)

        return net


print('Initializing environment')
x_max = y_max = 100
env = Environment(15, 10, 20, x_max, y_max)

print('Initializing vehicle')
vehicle = Vehicle(4, 4)

duration = 20000
dt = 150

# Create store for neuron membrane potentials
membrane_potentials = {}
for layer_index, layer in vehicle.net.layers.items():
    membrane_potentials[layer_index] = np.zeros([dt, layer.N])

T = np.arange(0, duration, dt)
x = np.zeros(len(T) + 1)
y = np.zeros(len(T) + 1)
w = np.zeros(len(T) + 1)
w[0] = np.pi / 4  # initial rotation

print('Preparing simulation')

# Draw Environment
plt.figure(2)
plt.xlim(0, x_max)
plt.ylim(0, y_max)
plt.title('Robot controlled by spiking neurons')
plt.xlabel('X')
plt.ylabel('Y')

for obj in env.objects:
    plt.scatter(obj['x'], obj['y'], s=np.pi * (obj['r'] ** 2), c='lime')

plt.ion()
plt.show()

print('Starting simulation')

for t in xrange(len(T)):

    sl, sr = env.read_sensors(x[t], y[t], w[t])

    # Reset the neuron firings tab, but carry over those firings that may not have yet reached their
    # target. Pull the time back to mark these firings as now negative.
    for layer_index, layer in vehicle.net.layers.items():
        firings = [[f[0] - dt, f[1]] for f in layer.firings if f[0] > dt - max_conduction_delay]
        layer.firings = np.array(firings)

    for t2 in xrange(dt):
        vehicle.net.layers[0].I = rn.poisson(sl * 15, vehicle.net.layers[0].N)
        vehicle.net.layers[1].I = rn.poisson(sr * 15, vehicle.net.layers[1].N)

        vehicle.net.layers[2].I = 5 * rn.randn(vehicle.net.layers[2].N)
        vehicle.net.layers[3].I = 5 * rn.randn(vehicle.net.layers[3].N)

        vehicle.net.tick(t2)

        for layer_index, layer in vehicle.net.layers.items():
            membrane_potentials[layer_index][t2, :] = layer.V

    for layer_index, layer in vehicle.net.layers.items():
        layer.firings = np.array(filter(lambda f: f[0] > 0, layer.firings))

    rl = 1.0 * len(vehicle.net.layers[2].firings) / dt / vehicle.net.layers[2].N * 1000
    rr = 1.0 * len(vehicle.net.layers[3].firings) / dt / vehicle.net.layers[3].N * 1000

    ul = (wheel_velocity_min / wheel_velocity_max + rl / motor_max * (1 - wheel_velocity_min / wheel_velocity_max))
    ur = (wheel_velocity_min / wheel_velocity_max + rr / motor_max * (1 - wheel_velocity_min / wheel_velocity_max))

    vehicle.move(env, ul, ur)

    x[t + 1], y[t + 1], w[t + 1] = vehicle.x, vehicle.y, vehicle.w

    # Plot membrane potential
    plt.figure(1)
    plt.clf()

    plt.subplot(221)
    plt.plot(membrane_potentials[0])
    plt.subplot(221)
    plt.title('Left sensory neurons')
    plt.ylabel('Membrane potential (mV)')
    plt.ylim(-90, 40)

    plt.subplot(222)
    plt.plot(membrane_potentials[1])
    plt.title('Right sensory neurons')
    plt.ylim(-90, 40)

    plt.subplot(223)
    plt.plot(membrane_potentials[2])
    plt.title('Left motor neurons')
    plt.ylabel('Membrane potential (mV)')
    plt.ylim(-90, 40)
    plt.xlabel('Time (ms)')

    plt.subplot(224)
    plt.plot(membrane_potentials[3])
    plt.title('Right motor neurons')
    plt.ylim(-90, 40)
    plt.xlabel('Time (ms)')

    plt.draw()

    plt.figure(2)
    plt.scatter(x, y, marker='.')
    plt.draw()

    # Pause for screen refresh
    plt.pause(.1)
