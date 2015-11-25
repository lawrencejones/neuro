import numpy as np


class NeuronNetworkLayer(object):

    """
    General model of a neuron network layer. Can be used in a more general NeuronNetwork.
    """

    def __init__(self, n):
        self.N = n
        self.I = np.zeros(n)

        # S[n][i,j] is the strength of connection from neuron j in layer n to neuron i in this layer
        self.S = {}

        # delay[n][i,j] is the delay between neuron j in layer n to neuron i in this layer
        self.delay = {}
        self.factor = {}

        self.firings = np.empty((0, 2))  # empty stack of tuples
        self.V = -65 * np.random.random_sample(n)  # random starting voltages

    def tick(self, dt, t):
        """
        Simulates a single ms of time by interpolating the next values for membrane potentials of
        each neuron in the layer.
        """

        no_of_steps = int(1 / dt)

        for step in xrange(no_of_steps):
            self._step_membrane_potential(dt, t)

            fired_neurons = np.where(self.V >= self.fire_threshold)[0]
            self._reset_neurons(fired_neurons)

            for neuron_index in fired_neurons:
                self._register_neuron_fire(neuron_index, t)

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

    def _register_neuron_fire(self, neuron_index, t):
        """
        Adds an entry to self.firings to record that the neuron has fired, at time t
        """

        if len(self.firings) > 0:
            self.firings = np.vstack([self.firings, [t, neuron_index]])
        else:
            self.firings = np.array([[t, neuron_index]])

    def _reset_neurons(self, neuron_indexes):
        """
        Method to override for resetting neurons and tracking variables after fire
        """

        raise NotImplementedError("Requires implementing in subclass")

    def _step_membrane_potential(self, dt, t):
        """
        Uses the Euler method to compute the next V (membrane potential).
        The exact details of the equation will differ depending on the model.
        """

        raise NotImplementedError("Requires implementing in subclass")
