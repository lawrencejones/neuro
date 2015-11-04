from QuadraticLayer import QuadraticLayer


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
