import numpy as np

import sys
import os
path = os.getcwd()
sys.path.append(path)

from nn.convolution.models import Convolution

np.random.seed(42)

input_data = np.random.randn(2, 10, 10, 3)
generator = np.random.default_rng(42)

colv1 = Convolution(unit_filters=1, filter_size=(3, 3), strides=(1, 1), optimize=True)
colv1.np_rng = generator
colv1.init(input_data)

colv2 = Convolution(unit_filters=1, filter_size=(3, 3), strides=(1, 1), optimize=False)
colv2.np_rng = generator
colv2.init(input_data)
print((colv1.p['W'] == colv2.p['W']).all())

forward1 = colv1.forward(input_data)
forward2 = colv2.forward(input_data)
res = (forward1 == forward2)
print(res.all())

output_data = np.random.rand(*forward2.shape)
backward1 = colv1.backward(output_data)
backward2 = colv2.backward(output_data)
res = (backward1 == backward2)
print(res.all())

colv2.compute_gradients()
W2 = colv2.g['dW']
b2 = colv2.g['db']

colv1.compute_gradients()
W1 = colv1.g['dW']
b1 = colv1.g['db']

res = (W1 == W2)
print(res.all())

res = (b1 == b2)
print(res.all())

print(np.isclose(W1, W2).all())
print(np.isclose(b1, b2).all())
print(W1[0])
print(W2[0])
print(b1)
print(b2)