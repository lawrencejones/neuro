"""
Simulates Izhikevich's neuron model using the Runge-Kutta method.
Parameters for regular spiking, fast spiking and bursting
neurons extracted from:

http://www.izhikevich.org/publications/spikes.htm
"""

import numpy as np


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

    def __init__(self, params, dt=0.01):
        """
        params -- dictionary of a,b,c,d params
        dt -- step size for tick and simulation
        """
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']

        self.dt = dt
        self.v = -65  # default initial membrane potential
        self.u = -1  # default initial rest potential

    def simulate(self, duration=200):
        """
        Implements the simulate interface, to yield the membrane potential over many iterations of
        Runge-Kutta simulation.
        """

        steps = int(duration / self.dt)

        T = np.arange(0, duration, self.dt)
        V = np.zeros(steps)
        U = np.zeros(steps)

        for i in xrange(steps - 1):

            V[i] = self.v
            U[i] = self.u

            fired = self.tick(IzNeuron.BASE_CURRENT)

            if fired:
                V[i] = IzNeuron.DIRAC_SPIKE  # pulse for visualisation

        return T, V, U

    def tick(self, i):
        """
        Implements the tick interface, where the internal variables of the neuron are projected
        forward by a single tick. Returns True if the neruon has fired, False otherwise.
        """

        # Runge-Kutta method
        k1 = self.__f(self.v, self.u, i)
        k2 = self.__f(self.v + 0.5 * self.dt * k1, self.u, i)
        k3 = self.__f(self.v + 0.5 * self.dt * k2, self.u, i)
        k4 = self.__f(self.v + self.dt * k3, self.u, i)

        # New projected v and u values
        self.v = self.v + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.u = self.u + self.dt * (self.a * (self.b * self.v - self.u))

        if self.__above_threshold(self.v):
            self.v = self.c  # reset to resting potential
            self.u = self.u + self.d  # update recovery variable

            return True

        return False

    def __above_threshold(self, value):
        return value >= IzNeuron.THRESHOLD

    def __f(self, v, u, i):
        """
        Computes dv/dt for the values of v and u
        """
        return (0.04 * v ** 2) + (5 * v) + 140 - u + i
