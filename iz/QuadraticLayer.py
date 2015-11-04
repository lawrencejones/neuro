from NeuronNetworkLayer import NeuronNetworkLayer
import numpy.random as rn

R = 1.0
tau = 5
vr = -65
vc = -50
a = 0.2


class QuadraticLayer(NeuronNetworkLayer):

    def __init__(self, n, fire_threshold=30):
        super(QuadraticLayer, self).__init__(n)

        self.A = 0.19 + (0.02 * rn.rand(n))
        self.fire_threshold = fire_threshold

    def _reset_neuron(self, neuron_index):
        self.V[neuron_index] = vr

    def _step_membrane_potential(self, dt, t):
        return self.V + dt * ((self.A * (vr - self.V) * (vc - self.V) + R * self.I) / tau)
