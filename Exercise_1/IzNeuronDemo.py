"""
Computational Neurodynamics
Exercise 1

Simulates Izhikevich's neuron model using the Runge-Kutta method.
Parameters for regular spiking, fast spiking and bursting
neurons extracted from:

http://www.izhikevich.org/publications/spikes.htm

(C) Murray Shanahan et al, 2015
"""

import sys
import matplotlib.pyplot as plt
from iz.IzNeuron import IzNeuron, IZ_PARAMETERS

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("""
        Desc:  Plots Izhikevich neruon models, both u and v
        Usage: python IzNeuronDemo.py <type-of-neuron>
        Examples...

            python IzNeuronDemo.py fast
            python IzNeuronDemo.py burst
            python IzNeuronDemo.py regular

        """)

        exit(-1)

    type_of_neuron = sys.argv[1]
    T, V, U = IzNeuron(IZ_PARAMETERS[type_of_neuron], dt=0.01).simulate()

    # Plot the membrane potential
    plt.subplot(211)
    plt.plot(T, V)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential v (mV)')
    plt.title('Izhikevich Neuron')

    # Plot the reset variable
    plt.subplot(212)
    plt.plot(T, U)
    plt.xlabel('Time (ms)')
    plt.ylabel('Reset variable u')
    plt.show()
