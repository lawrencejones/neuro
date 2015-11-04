from QuadraticNetwork import QuadraticNetwork
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt


def create_two_layer_network(N0, N1):

    F = 50 / np.sqrt(N1)
    D = 5
    Dmax = 6

    net = QuadraticNetwork([N0, N1], Dmax)

    net.layers[1].S[0] = np.ones([N1, N0])
    net.layers[1].factor[0] = F
    net.layers[1].delay[0] = D * np.ones([N1, N0], dtype=int)

    return net


def add_dirac(V, firings, dirac_pulse=30):
    if firings.size > 0:
        V[firings[:, 0], firings[:, 1]] = dirac_pulse


N1, N2 = 4, 4
duration = 500
base_current = 20

net = create_two_layer_network(N1, N2)

V1 = np.zeros([duration, N1])
V2 = np.zeros([duration, N2])

for t in xrange(duration):

    net.layers[0].I = base_current * np.ones(N1)
    net.layers[1].I = np.zeros(N2)

    net.tick(t)

    V1[t] = net.layers[0].V
    V2[t] = net.layers[1].V

add_dirac(V1, net.layers[0].firings)
add_dirac(V2, net.layers[1].firings)

# Plot membrane potentials
plt.figure(1)
plt.subplot(211)
plt.plot(range(duration), V1)
plt.title('Population 1 membrane potentials')
plt.ylabel('Voltage (mV)')
plt.ylim([-90, 40])

plt.subplot(212)
plt.plot(range(duration), V2)
plt.title('Population 2 membrane potentials')
plt.ylabel('Voltage (mV)')
plt.ylim([-90, 40])
plt.xlabel('Time (ms)')

# Raster plots of firings
plt.figure(3)
plt.subplot(211)
plt.scatter(net.layers[0].firings[:, 0], net.layers[0].firings[:, 1] + 1, marker='.')
plt.xlim(0, duration)
plt.ylabel('Neuron number')
plt.ylim(0, N1 + 1)
plt.title('Population 1 firings')

plt.subplot(212)
plt.scatter(net.layers[1].firings[:, 0], net.layers[1].firings[:, 1] + 1, marker='.')
plt.xlim(0, duration)
plt.ylabel('Neuron number')
plt.ylim(0, N2 + 1)
plt.xlabel('Time (ms)')
plt.title('Population 2 firings')

plt.show()
