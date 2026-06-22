# mygrad

a from-scratch autograd engine and neural network library. Based off Karpathy's micrograd. 

### What is mygrad?

mygrad implements a lightweight automatic differentation (autograd) engine that allows you to compute gradients of arbitrary computational graphs. It also has a simple neural network library on top to demo how autodiff powers deep learning.

I built this as a learning milestone while studying deep learning fundamentals, so the code may not be the cleanest and most optimized.

**features:**
- scalar-valued autograd engine w/ backpropagation
- support for basic operations (+, -, /, **)
- activation functions (tanh, exp)
- Neural network components (Neuron, Layer, MLP)
- computational graph visualization

### Installation

```bash
git clone https://github.com/randomchaffee/mygrad.git
cd mygrad

# deps
pip install -r requirements.txt
```

### Quick start sample

```python
from mygrad import Value, MLP

# create a simple 2-layer MLP: 3 inputs -> 4 hidden layers -> 1 output

# forward pass
inputs = [2.0, 3.0, -1.0]
output = model(inputs)

# backward pass
output.backward()

# access gradients
for i, param in enumerate(model.parameters()):
    print(f"Parameter {i} gradient: {param.grad}")
```

### How it works
*Automatic Differentation (Autograd)*: Every operation you perform creates a computational graph. The `backward()` method traverses this graph in reverse, computing gradients using the chain rule.

*Neural Networks*: The network is composed of layers of neurons. Each neuron computes `tanh(w * X + b)`. DUring training, gradients flow backward to update the weights via what we call gradient descent.

### File Organization

- mygrad/value.py - core class implementing the autograd engine
- mygrad/nn.py - neural network components
- mygrad/visualizaton.py - computational graph visualization tools

### Resources

- [Karpathy's micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0)
