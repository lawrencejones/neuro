from QuadraticNetwork import QuadraticLayer
import numpy as np
import matplotlib.pyplot as plt


def add_dirac(V, firings, dirac_pulse=30):
    if firings.size > 0:
        V[firings[:, 0], firings[:, 1]] = dirac_pulse


N1 = 2
duration = 50
dt = 0.02
base_current = 20
steps = int(duration / dt)

layer = QuadraticLayer(N1)
layer.I = base_current * np.ones(N1)

V = np.zeros([duration, N1])

for t in xrange(duration):
    for step in xrange(int(1 / dt)):
        layer.tick(dt, t)
    V[t] = layer.V

add_dirac(V, layer.firings)

# Plot membrane potentials
plt.figure(1)
plt.subplot(211)
plt.plot(range(duration), V)
plt.title('Membrane potentials')
plt.ylabel('Voltage (mV)')
plt.ylim([-90, 40])

plt.show()
