import numpy as np
import matplotlib.pyplot as plt

GNa = 120.0
ENa = 115.0

GK = 36.0
EK = -12.0

GL = 0.3
EL = 10.6


def dvdt(v, m, n, h, base_current):
    sodium = GNa * (m**3) * h * (v - ENa)
    potassium = GK * (n**4) * (v - EK)
    leakage = GL * (v - EL)

    return base_current - (sodium + potassium + leakage)


def dmdt(v, m):
    alpha = (2.5 - 0.1 * v) / (np.exp(2.5 - 0.1 * v) - 1)
    beta = 4.0 * np.exp(-v / 18)

    return (alpha * (1 - m)) - (beta * m)


def dndt(v, n):
    alpha = (0.1 - 0.01 * v) / (np.exp(1.0 - 0.1 * v) - 1)
    beta = 0.125 * np.exp(-v / 80.0)

    return (alpha * (1 - n)) - (beta * n)


def dhdt(v, h):
    alpha = (0.07 * np.exp(-v / 20.0))
    beta = 1.0 / (np.exp(3.0 - 0.1 * v) + 1.0)

    return (alpha * (1 - h)) - (beta * h)


class HodgkinHuxleyNeuron(object):

    def __init__(self, base_current=10, v0=-10):
        self.base_current = base_current
        self.v0 = v0

    def simulate(self, duration=100, dt=0.01):

        steps = int(duration / dt)

        T = np.arange(0, duration, dt)

        V = np.zeros(steps)
        M = np.zeros(steps)
        N = np.zeros(steps)
        H = np.zeros(steps)

        V[0] = self.v0

        for i in xrange(steps - 1):
            M[i + 1] = M[i] + dt * dmdt(V[i], M[i])
            N[i + 1] = N[i] + dt * dndt(V[i], N[i])
            H[i + 1] = H[i] + dt * dhdt(V[i], H[i])

            V[i + 1] = V[i] + dt * dvdt(V[i], M[i], N[i], H[i], self.base_current)

        return T, V, M, N, H


if __name__ == '__main__':

    T, V, M, N, H = HodgkinHuxleyNeuron().simulate()

    plt.title('Hodgkin-Huxley Neuron Model')

    plt.subplot(211)
    plt.ylabel('Membrane potential v (mV)')
    plt.xlabel('Time (ms)')
    plt.plot(T, V)

    plt.subplot(212)
    plt.ylabel('m')
    plt.xlabel('Time (ms)')
    plt.plot(T, M)

    plt.subplot(212)
    plt.ylabel('n')
    plt.xlabel('Time (ms)')
    plt.plot(T, N)

    plt.subplot(212)
    plt.ylabel('h')
    plt.xlabel('Time (ms)')
    plt.plot(T, H)

    plt.show()

