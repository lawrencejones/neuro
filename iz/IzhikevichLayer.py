from NeuronNetworkLayer import NeuronNetworkLayer
import numpy as np
import numpy.random as rn

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


def randomize_params(base, n, epsilon=0.1):

    """
    Given a base parameter, produces an array of N length of that parameter randomized
    within a 10% variance of the original value
    """

    return (base - (0.5 * epsilon * base)) + (epsilon * base * rn.rand(n))


def dvdt(V, U, I):

    """
    Computes dv/dt for the array values of V U and I
    """

    return (0.04 * V ** 2) + (5 * V) + 140 - U + I


class IzhikevichLayer(NeuronNetworkLayer):

    """
    Implements an Izhekevich neuron model.
    """

    def __init__(self, n, fire_threshold=30, fire_type='regular'):
        super(IzhikevichLayer, self).__init__(n)

        self.A = randomize_params(IZ_PARAMETERS[fire_type]['a'], n)
        self.B = randomize_params(IZ_PARAMETERS[fire_type]['b'], n)
        self.C = randomize_params(IZ_PARAMETERS[fire_type]['c'], n)
        self.D = randomize_params(IZ_PARAMETERS[fire_type]['d'], n)

        self.U = -1 * np.ones(n)

        self.fire_threshold = fire_threshold

    def _reset_neurons(self, neuron_indexes):
        self.V[neuron_indexes] = self.C[neuron_indexes]
        self.U[neuron_indexes] = self.U[neuron_indexes] + self.D[neuron_indexes]

    def _step_membrane_potential(self, dt, t):

        # Runge-Kutta method
        k1 = dvdt(self.V, self.U, self.I)
        k2 = dvdt(self.V + 0.5 * dt * k1, self.U, self.I)
        k3 = dvdt(self.V + 0.5 * dt * k2, self.U, self.I)
        k4 = dvdt(self.V + dt * k3, self.U, self.I)

        # New projected v and u values
        self.V = self.V + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.U = self.U + dt * (self.A * (self.B * self.V - self.U))

        return self.V
