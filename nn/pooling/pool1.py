import numpy as np

import sys
import os
path = os.getcwd()
sys.path.append(path)

def optimized_pooling_forward(layer, A):
    """Optimized forward propagation for the pooling layer."""
    layer.fc['X'] = A
    f_h, f_w = layer.d['ph'], layer.d['pw']
    stride_h, stride_w = layer.d['sh'], layer.d['sw']
    Z = np.zeros((layer.d['m'], layer.d['oh'], layer.d['ow'], layer.d['d']))

    # Pooling operation
    for i in range(layer.d['oh']):
        for j in range(layer.d['ow']):
            h_start, h_end = i * stride_h, i * stride_h + f_h
            w_start, w_end = j * stride_w, j * stride_w + f_w
            A_slice = A[:, h_start : h_end, w_start : w_end, :]
            Z[:, i, j, :] = np.max(A_slice, axis=(1, 2))

    A = layer.fc['A'] = layer.fc['Z'] = Z

    return A  

    layer.fc['A_prev'] = A  # Cache for backward pass
    return Z

def optimized_pooling_backward(layer, dX):
    """Optimized backward propagation for the pooling layer."""
    layer.bc['dA'] = dX
    # A_prev = layer.fc['A']  # From forward pass
    # m, layer.d['h'], layer.d['w'], d = A_prev.shape
    f_h, f_w = layer.d['ph'], layer.d['pw']
    stride_h, stride_w = layer.d['sh'], layer.d['sw']
    # layer.d['oh'], layer.d['ow'] = dX.shape[1:3]
    dZ = dX
    # Initialize gradient dA_prev
    dX = np.zeros_like(layer.fc['X'])

    for i in range(layer.d['oh']):
        for j in range(layer.d['ow']):
            h_start, h_end = i * stride_h, i * stride_h + f_h
            w_start, w_end = j * stride_w, j * stride_w + f_w

            # Slice A_prev
            A_slice = layer.fc['X'][:, h_start:h_end, w_start : w_end, :]

            # Create mask from A_slice to identify max positions
            mask = A_slice == np.max(A_slice, axis=(1, 2), keepdims=True)

            # Backpropagate only through the max positions
            dX[:, h_start:h_end, w_start:w_end, :] += mask * dZ[:, i, j, :][:, np.newaxis, np.newaxis, :]

    layer.bc['dX'] = dX

    return dX

from nn.pooling.models import Pooling

# np.random.seed(42)
input_data = np.random.rand(64, 1000, 1000, 3)
def debug(input):
    for i in range(input.shape[1]):
        for j in range(input.shape[2]):
            print(input[0][i][j], end = " ")
        print()

#debug(input_data)
pool1 = Pooling(pool_size=(5, 5), strides=(1, 1))

pool1.compute_shapes(input_data)

from memory_profiler import profile

@profile
def run_memory_test():
    forward1 = optimized_pooling_forward(pool1, input_data)

    output_data = np.random.rand(*forward1.shape)
    backward1 = optimized_pooling_backward(pool1, output_data)

if __name__ == "__main__":
    run_memory_test()