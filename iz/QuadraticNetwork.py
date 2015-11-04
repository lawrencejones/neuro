import numpy as np
import numpy.random as rn

R = 1.0
tau = 5
vr = -65
vc = -50
a = 0.2
threshold = 30


class QuadraticLayer(object):

    def __init__(self, n):
        self.N = n
        self.I = np.zeros(n)
        self.V = np.zeros(n)

        # S[n] is connectivity matrix from layer n to this layer
        # S[n][i,j] is the strength of connection from neuron j in layer n to neuron i in this layer
        self.S = {}

        # delay[n] is the delay matrix from layer n to this layer
        # delay[n][i,j] is the delay between neuron j in layer n to neuron i in this layer
        self.delay = {}
        self.factor = {}

        self.V = vr * np.ones(n)
        self.A = 0.19 + (0.02 * rn.rand(n))
        self.firings = np.array([])

    def firings_after(self, cutoff):
        """
        Returns the firing events that happened after the cutoff in steps
        """
        index = len(self.firings) - 1

        while index > 0:
            firing_time, neuron_index = self.firings[index]
            if firing_time < cutoff:
                return
            yield self.firings[index]
            index = index - 1

    def tick(self, dt, t):
        for k in xrange(int(1 / dt)):
            self.V = self.V + dt * ((self.A * (vr - self.V) * (vc - self.V) + R * self.I) / tau)

        fired_neurons = np.where(self.V >= threshold)[0]

        for f in fired_neurons:
            if len(self.firings) > 0:
                self.firings = np.vstack([self.firings, [t, f]])
            else:
                self.firings = np.array([[t, f]])

            self.V[f] = vr  # reset neuron


class QuadraticNetwork(object):

    def __init__(self, neurons_per_layer, d_max):
        self.neurons_per_layer = neurons_per_layer
        self.d_max = d_max

        self.layers = [QuadraticLayer(n) for n in neurons_per_layer]
        self.t = 0
        self.dt = 0.02

    def tick(self, t):
        for layer_index in xrange(len(self.layers)):
            self.tick_layer(layer_index, t)

    def tick_layer(self, layer_index, t):
        layer = self.layers[layer_index]

        for input_layer_index, S in layer.S.iteritems():
            input_layer = self.layers[input_layer_index]

            delay = layer.delay[input_layer_index]
            scaling_factor = layer.factor[input_layer_index]

            for (firing_time, neuron_index) in input_layer.firings_after(t - self.d_max):
                idx = delay[:, neuron_index] == (t - firing_time)
                layer.I[idx] += scaling_factor * S[idx, neuron_index]

        layer.tick(self.dt, t)
