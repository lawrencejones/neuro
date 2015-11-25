# Neuron

Running out of time, so this will be very brief.

As I've gone through the tutorials, I've rewritten most of your code so that I could learn how it
works. I've used my own neuro libraries to produce this coursework as a result.

Please run `python Q1.py <rewiring_probability> <duration_in_ms>` to kick off the simulation.

Example runs...

`python Q1.py .2 1000  # p=0.2, will run for the full 1000ms`


If the code seems a bit scrappy, it's because I was really rushing after debugging all day. My
apologies for this.

All requested plots are in the `q1_graphics` directory.

## Structure

I have placed all heavy code into the `neuro` directory, and just use the `Q1.py` file to instrument
those modules.

```
neuro
├── IzhikevichLayer.py               # simulates a layer of neurons via the Izhikevich model
├── ModularFocalNetwork.py           # produces connectivity matrix that has focal intra-module connections
├── ModularSmallWorldNetwork.py      # produces connectivity matrix with modules that can rewire itself
├── NetworkSimulator.py              # runs a simulation over a NeuronNetwork
├── NeuronNetwork.py                 # manages the simulation of the neuron network
├── NeuronNetworkLayer.py            # base class for specific model layers, see IzhikevichLayer
├── Plotters.py                      # plotting utility methods for display
├── __init__.py
```

