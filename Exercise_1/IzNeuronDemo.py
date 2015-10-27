"""
Computational Neurodynamics
Exercise 1

Simulates Izhikevich's neuron model using the Euler method.
Parameters for regular spiking, fast spiking and bursting
neurons extracted from:

http://www.izhikevich.org/publications/spikes.htm

(C) Murray Shanahan et al, 2015
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


IZ_PARAMETERS = {
    'regular': {
        'a': 0.02,
        'b': 0.2,
        'c': -65,
        'd': 8,
    },
    'fast': {
        'a': 0.02,
        'b': 0.25,
        'c': -65,
        'd': 2,
    },
    'burst': {
        'a': 0.02,
        'b': 0.2,
        'c': -50,
        'd': 2,
    },
}


class IzNeuron(object):

    BASE_CURRENT = 10
    THRESHOLD = 30
    DIRAC_SPIKE = 30

    def __init__(self, params):
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']

    def simulate(self, v0=-65, u0=-1, duration=200, dt=0.01, I=10):

        steps = int(duration / dt)

        T = np.arange(0, duration, dt)
        V = np.zeros(steps)
        U = np.zeros(steps)

        V[0] = v0
        U[0] = u0

        for i in xrange(steps - 1):
            V[i + 1] = V[i] + dt * (0.04 * V[i]**2 + 5 * V[i] + 140 - U[i] + IzNeuron.BASE_CURRENT)
            U[i + 1] = U[i] + dt * (self.a * (self.b * V[i] - U[i]))

            if self.__above_threshold(V[i + 1]):
                V[i] = IzNeuron.DIRAC_SPIKE  # pulse for visualisation
                V[i + 1] = self.c  # reset to resting potential
                U[i + 1] = U[i + 1] + self.d  # update recovery variable

        return T, U, V

    def __above_threshold(self, value):
        return value >= IzNeuron.THRESHOLD


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("""
        Desc:  Plots Izhikevich neruon models, both u and v
        Usage: python IzNeuronDemo.py <type-of-neuron>
        Examples...

            python IzNeuronDemo.py fast
            python IzNeuronDemo.py burst
            python IzNeuronDemo.py regular

        """)

        exit(-1)

    type_of_neuron = sys.argv[1]
    T, U, V = IzNeuron(IZ_PARAMETERS[type_of_neuron]).simulate()

    # Plot the membrane potential
    plt.subplot(211)
    plt.plot(T, V)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential v (mV)')
    plt.title('Izhikevich Neuron')

    # Plot the reset variable
    plt.subplot(212)
    plt.plot(T, U)
    plt.xlabel('Time (ms)')
    plt.ylabel('Reset variable u')
    plt.show()
