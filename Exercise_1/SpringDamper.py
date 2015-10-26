import numpy as np
import matplotlib.pyplot as plt


def accelleration(v, y):
    return (-1 / m) * (c * v + k * y)

m = 1
c = 0.1
k = 1

duration = 100
dt = 0.001
n = duration / dt

Y = np.zeros(n)
V = np.zeros(n)
A = np.zeros(n)

Y[0] = 1
V[0] = 0
A[0] = accelleration(V[0], Y[0])

for i in xrange(1, int(duration / dt)):
    Y[i] = Y[i-1] + dt * V[i-1]
    V[i] = V[i-1] + dt * A[i-1]
    A[i] = accelleration(V[i], Y[i])


plt.plot(np.arange(0, duration, dt), Y, 'y', label='Euler method for Y')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()
