# Neuron Models

Run the `NetworkDemo.py` file to see the behaviour of networks under different models.

Example calls...

```sh
# Simulate an Izhikevich neuron model with 2 network layers of 8 and 4 neurons respectively,
# for a duration of 500ms, with a base current of 10
python iz/NetworkDemo.py --model izhikevich --base-current 10 --duration 500 8 4

# Simulate an QIF neuron model with 2 network layers of 20 and 10 neurons respectively,
# for a duration of 200ms, with a base current of 20
python iz/NetworkDemo.py --model quadratic --base-current 20 --duration 200 20 10
```
