import numpy as np
from NeuronNetwork import add_dirac_pulse


def simulate(net, duration, base_current=0, bg_lam=0, bg_scale=0):
    """
    Simulates the given network in action for the given duration, graphing the neuron spikes of the
    different layers.
    """

    VS = [np.zeros([duration, net.layers[idx].N]) for idx in range(len(net.layers))]

    for t in xrange(duration):

        # Progress marker
        if t % 100 == 0:
            print(t)

        net.layers[0].I = base_current * np.ones(net.layers[0].N)
        if (bg_lam > 0) and (bg_scale != 0):
            random_spikes = np.where(np.random.poisson(lam=bg_lam, size=net.layers[0].N) > 0)[0]
            net.layers[0].I[random_spikes] += bg_scale

        for layer_index in range(1, len(net.layers)):
            net.layers[layer_index].I = np.zeros(net.layers[layer_index].N)

        net.tick(t)

        for layer_index in range(len(net.layers)):
            VS[layer_index][t] = net.layers[layer_index].V

    for layer_index, layer in net.layers.items():
        add_dirac_pulse(VS[layer_index], layer.firings)

    return VS, net
