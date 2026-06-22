import numpy as np
import matplotlib.pyplot as plt
import random
# %matplotlib inline

from .value import Value

# building a neural net

class Neuron:
    # the constructor will take number of inputs to the neuron
    # and create a weight that is rand n between -1 and 1 for every input
    # and a bias that controls that controls the overall trigger happiness of the neuron
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        # we need to multiply all the elements of w with all elements of x pairwise

        # zip takes two iterators and creates a new iterator that iterates over the
        # tuples of their corresponding entries
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # activation
        out = act.tanh()
        return out

    # collects paramaters of the neural net all in one array
    def parameters(self):
        return self.w + [self.b]

# what is a layer of neurons?
# its just a set of neurons evaluated independently
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # identical to above
        # params = []
        # for neuron in self.neurons:
        #     ps neuron.parameters()
        #     params.extend(ps)
        # return params

# in an MLP, layers just feed into each other sequentially (see visualizations online)
class MLP:
    # we take the number of inputs
    # but instead of taking a single nout (number of neurons in a single layer)
    # we take a list of nouts and this listifies the sizes of all the layers that we want in our MLP
    def __init__(self, nin, nouts):
        # we put them all together
        sz = [nin] + nouts
        # iterate over consecutive pairs of these sizes and create layer objects for them
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    # then we call them sequentially
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]