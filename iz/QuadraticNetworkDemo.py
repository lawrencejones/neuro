from NeuronNetwork import NeuronNetwork, add_dirac_pulse
from QuadraticLayer import QuadraticLayer
from Plotters import plot_show, plot_membrane_potentials, plot_firings

import numpy as np


N1, N2 = 20, 20
duration = 100
base_current = 20

net = NeuronNetwork.create_feed_forward(QuadraticLayer, [N1, N2], scaling_factor=50)

V1 = np.zeros([duration, N1])
V2 = np.zeros([duration, N2])

for t in xrange(duration):

    net.layers[0].I = base_current * np.ones(N1)
    net.layers[1].I = np.zeros(N2)

    net.tick(t)

    V1[t] = net.layers[0].V
    V2[t] = net.layers[1].V

add_dirac_pulse(V1, net.layers[0].firings)
add_dirac_pulse(V2, net.layers[1].firings)

plot_membrane_potentials([V1, V2], duration, 1)
plot_firings(net, duration, 2)

plot_show()
