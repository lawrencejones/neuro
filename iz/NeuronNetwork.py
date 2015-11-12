import numpy as np
from NeuronNetworkLayer import NeuronNetworkLayer


def add_dirac_pulse(V, firings, dirac_pulse=30):
    """
    Given a record of when neurons fired, will add Dirac pulses of the given magnitude to the
    membrane potentials in V.
    """

    if firings.size > 0:
        V[firings[:, 0], firings[:, 1]] = dirac_pulse


class NeuronNetwork(object):

    """
    Generalised model of a neuron network. Controls interaction between various neuron layers.
    """

    @classmethod
    def create_feed_forward(cls, neuron_layer_class, no_of_neurons_in_layers, scaling_factor):

        layers = [neuron_layer_class(n) for n in no_of_neurons_in_layers]
        net = cls(layers)

        for layer_index in range(1, len(layers)):
            from_layer, to_layer = layers[layer_index - 1:layer_index + 1]
            layer_scaling_factor = scaling_factor / np.sqrt(from_layer.N)

            net.connect_layers([from_layer, to_layer], scaling_factor=layer_scaling_factor)

        return net

    def __init__(self, neuron_layers, d_max=5):
        """
        Initialises the neuron network with given layers.

        neuron_layers -- collection of neuron layer objects (Quadratic, HodgkinHuxley, Izkevich)
        """

        self.layers = dict(zip(range(len(neuron_layers)), neuron_layers))

        self.dt = 0.2
        self.d_max = d_max

    def tick(self, t):
        """
        Advances the simulation by one step on all layers
        """

        for layer_index in xrange(len(self.layers)):
            self.tick_layer(layer_index, t)

    def connect_layers(self, layers, scaling_factor=None, delay=5, S=None):
        """
        Connects the configured layers with the following parameters...

        layers -- an array of layer_indexes in self.layers
        scaling_factor -- multiplier for current from spiking neurons to the destination neurons
        delay -- conduction delay, ms after a neuron fires from layer and is picked up in to layer
        S -- synaptic connection strength matrix, by default all-to-all
        """

        if len(layers) != 2:
            raise StandardError("Expected layers to be an array of [from_index,to_index]")

        from_layer_index, from_layer = self._identify_layer(layers[0])
        to_layer_index, to_layer = self._identify_layer(layers[1])

        # Create connectivity matrix if not supplied, assume full connectivity
        to_layer.S[from_layer_index] = \
            np.ones([to_layer.N, from_layer.N], dtype=int) if S is None else S

        # Create scaling factor for each connection between the layers, default is sqrt(N1)
        to_layer.factor[from_layer_index] = \
            np.sqrt(to_layer.N) if scaling_factor is None else scaling_factor

        # The conductive delay between neurons in from->to
        to_layer.delay[from_layer_index] = delay * np.ones([to_layer.N, from_layer.N], dtype=int)

        return self

    def tick_layer(self, layer_index, t):
        """
        Advances a specific layer by one tick in the simulation.

        layer_index -- the index of the layer to tick
        t -- the current sim time, for marking fired neurons
        """

        layer = self.layers[layer_index]

        for input_layer_index, S in layer.S.iteritems():
            input_layer = self.layers[input_layer_index]

            delay = layer.delay[input_layer_index]
            scaling_factor = layer.factor[input_layer_index]

            for (firing_time, neuron_index) in input_layer.firings_after(t - self.d_max):
                idx = delay[:, neuron_index] == (t - firing_time)
                layer.I[idx] += scaling_factor * S[idx, neuron_index]

        layer.tick(self.dt, t)

    def _identify_layer(self, layer_item):
        """
        Receives either a layer index, or a layer and returns layer_index,layer
        """

        if isinstance(layer_item, NeuronNetworkLayer):
            return [(index, layer_item) for index, _ in self.layers.items() if _ == layer_item][0]
        else:
            return (layer_item, self.layers[layer_item])

