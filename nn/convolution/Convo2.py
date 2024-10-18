import numpy as np

import sys
import os
path = os.getcwd()
sys.path.append(path)

from nn.commons.io import padding 
from nn.convolution.backward import initialize_backward
from nn.convolution.forward import initialize_forward

from memory_profiler import profile

def optimized_convolution_forward(layer, A):
    """Forward propagate signal to next layer."""
    # (1) Initialize cache and pad image
    X = initialize_forward(layer, A)  # (m, h, w, d)

    # (2) Create empty output Z based on output dimensions
    Z = np.zeros((X.shape[0], layer.d['oh'], layer.d['ow'], layer.d['u']))

    # (3) Iterate over output height and width
    for oh in range(layer.d['oh']):
        for ow in range(layer.d['ow']):
            # Calculate slice indices based on strides
            h_start = oh * layer.d['sh']
            h_end = h_start + layer.d['fh']
            w_start = ow * layer.d['sw']
            w_end = w_start + layer.d['fw']

            # Slice the input image
            X_slice = X[:, h_start:h_end, w_start:w_end, :]

            # Perform the convolution (element-wise multiplication and summation)
            Z[:, oh, ow, :] = np.sum(X_slice[:, :, :, :, None] * layer.p['W'], axis=(1, 2, 3))

    # (4) Add bias
    Z += layer.p['b']

    # (5) Non-linear activation
    A = layer.fc['A'] = layer.activate(Z)
    layer.fc['Z'] = Z  # Cache Z for backward pass

    return A  # Output to next layer

def optimized_convolution_backward(layer, dX):
    """Backward propagate error gradients to previous layer."""
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)  # (m, oh, ow, u)

    # (2) Compute gradient of loss with respect to Z
    dZ = layer.bc['dZ'] = dA * layer.activate(layer.fc['Z'], deriv=True)

    # (3) Initialize backward output dL/dX
    dX = np.zeros_like(layer.fc['X'])  # (m, h, w, d)

    # (4) Iterate over output height and width
    for oh in range(layer.d['oh']):
        for ow in range(layer.d['ow']):
            # Calculate slice indices based on strides
            h_start = oh * layer.d['sh']
            h_end = h_start + layer.d['fh']
            w_start = ow * layer.d['sw']
            w_end = w_start + layer.d['fw']

            # Extract the gradient slice for this output pixel
            dZ_slice = dZ[:, oh, ow, :]  # (m, u)

            # (5) Multiply with the filter weights and accumulate the gradient
            for c in range(layer.d['d']):  # Iterate over channels
                dX[:, h_start:h_end, w_start:w_end, c] += np.sum(
                    dZ_slice[:, None, None, :] * layer.p['W'][:, :, c, :], axis=-1
                )

    layer.bc['dX'] = dX

    return dX

from nn.convolution.models import Convolution
#input_data = np.random.randn(64, 224, 224, 3)
input_data = np.random.randn(64, 300, 300, 3)
generator = np.random.default_rng(42)

pool2 = Convolution(unit_filters=32, filter_size=(3, 3), strides=(1, 1))
pool2.np_rng = generator
pool2.init(input_data)

from nn.pooling.models import Pooling
pool = Pooling(pool_size=(3, 3), strides = (1, 1))

@profile
def run_memory_test():
    forward2 = pool2.forward(input_data)
    output_data = np.random.rand(*forward2.shape)
    backward2 = pool2.backward(output_data)
    pool.forward(output_data)

if __name__ == "__main__":
    run_memory_test()