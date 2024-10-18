# nn/nn/convolution/parameters.py
# Standard library imports
import math

# Related third party imports
import numpy as np


def convolution_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, h, w, d)

    layer.d['m'] = layer.fs['X'][0]     # Number of samples  (m)
    layer.d['h'] = layer.fs['X'][1]     # Height of features (h)
    layer.d['w'] = layer.fs['X'][2]     # Width of features  (w)
    layer.d['d'] = layer.fs['X'][3]     # Depth of features  (d)

    # Output height (oh) and width (ow)
    layer.d['oh'] = math.floor((layer.d['h']-layer.d['fh']) / layer.d['sh']) + 1
    layer.d['ow'] = math.floor((layer.d['w']-layer.d['fw']) / layer.d['sw']) + 1

    # Shapes for trainable parameters
    # filter_height (fh), filter_width (fw), features_depth (d), unit_filters (u)
    layer.fs['W'] = (layer.d['fh'], layer.d['fw'], layer.d['d'], layer.d['u'])
    layer.fs['b'] = (layer.d['u'], )

    return None


def convolution_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # For linear activation of inputs (Z)
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    # np.random.seed(42)
    # layer.p['W'] = np.random.randn(*layer.fs['W'])
    layer.p['b'] = np.zeros(layer.fs['b']) # Z = X * W + b

    return None


def convolution_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # Gradients initialization with respect to parameters
    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    Xb = layer.fc['Xb']     # Input blocks of forward propagation
    dZ = layer.bc['dZ']     # Gradient of the loss with respect to Z

    # Expand dZ dimensions with respect to Xb
    dZb = dZ
    dZb = np.expand_dims(dZb, axis=3)    # (m, oh, ow, 1, u)
    dZb = np.expand_dims(dZb, axis=3)    # (m, oh, ow, 1, 1, u)
    dZb = np.expand_dims(dZb, axis=3)    # (m, oh, ow, 1, 1, 1, u)

    # (1) Gradient of the loss with respect to W, b
    dW = layer.g['dW'] = np.sum(dZb * Xb, axis=(2, 1, 0))   # (1.1) dL/dW
    db = layer.g['db'] = np.sum(dZb, axis=(2, 1, 0))        # (1.2) dL/db

    layer.g['db'] = db.squeeze() if layer.use_bias else 0.

    return None

def optimized_convolution_compute_gradients(layer):
    """Compute gradients with respect to weights and biases for convolution layer."""

    # Initialize gradients for weights and biases
    dW = np.zeros_like(layer.p['W'])  # Gradient w.r.t. weights
    db = np.zeros_like(layer.p['b'])  # Gradient w.r.t. bias
    
    X = layer.fc['X']  # Input to the convolution layer
    dZ = layer.bc['dZ']  # Gradient of the loss w.r.t. Z (from backward pass)

    m = layer.d['m']  # Batch size

    # (1) Iterate over output height and width
    for oh in range(layer.d['oh']):
        for ow in range(layer.d['ow']):
            # Calculate slice indices based on strides
            h_start = oh * layer.d['sh']
            h_end = h_start + layer.d['fh']
            w_start = ow * layer.d['sw']
            w_end = w_start + layer.d['fw']

            # Slice the input corresponding to this output position
            X_slice = X[:, h_start:h_end, w_start:w_end, :]  # Input slice
            dZ_slice = dZ[:, oh, ow, :]  # Gradient slice for current output

            # (2) Compute gradients for the weights (sum over minibatch and spatial dimensions)
            for c in range(layer.d['d']):  # Iterate over input channels
                for u in range(layer.d['u']):  # Iterate over output channels
                    dW[:, :, c, u] += np.sum(X_slice[:, :, :, c] * dZ_slice[:, u][:, None, None], axis=0)

            # (3) Compute gradients for the bias
            db += np.sum(dZ_slice, axis=0)

    # Normalize by batch size
    layer.g['dW'] = dW  # Normalize weights gradient
    layer.g['db'] = db if layer.use_bias else 0.

    return None


def convolution_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
