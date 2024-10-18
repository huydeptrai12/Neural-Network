# nn/nn/convolution/forward.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nn.commons.io import padding


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of convolution layer.
    :type layer: :class:`nn.convolution.models.Convolution`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Input blocks of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = padding(A, layer.d['p'])

    return X


def convolution_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache and pad image
    layer.fc['X'] = padding(A, layer.d['p'])  # (m, h, w, d)

    # (2) Slice input w.r.t. filter size (fh, fw) and strides (sh, sw)
    layer.fc['Xb'] = np.array([[layer.fc['X'][ :, h:h + layer.d['fh'], w:w + layer.d['fw'], :]
                    # Inner loop
                    # (m, h, w, d) ->
                    # (ow, m, h, fw, d)
                    for w in range(layer.d['w'] - layer.d['fw'] + 1)
                    if w % layer.d['sw'] == 0]
                # Outer loop
                # (ow, m, h, fw, d) ->
                # (oh, ow, m, fh, fw, d)
                for h in range(layer.d['h'] - layer.d['fh'] + 1)
                if h % layer.d['sh'] == 0])

    # (3) Bring back m along axis 0
    layer.fc['Xb'] = np.moveaxis(layer.fc['Xb'], 2, 0)
    # (oh, ow, m, fh, fw, d) ->
    # (m, oh, ow, fh, fw, d)

    # (4) Add dimension for filter units (u) on axis 6
    layer.fc['Xb'] = np.expand_dims(layer.fc['Xb'], axis=6)
    # (m, oh, ow, fh, fw, d) ->
    # (m, oh, ow, fh, fw, d, 1)

    # (5.1) Linear activation Xb -> Zb
    layer.fc['Z'] = layer.fc['Xb'] * layer.p['W']
    # (m, oh, ow, fh, fw, d, 1) - Xb
    #            (fh, fw, d, u) - W

    # (5.2) Sum block products
    layer.fc['Z'] = np.sum(layer.fc['Z'], axis=(5, 4, 3))
    # (m, oh, ow, fh, fw, d, u) - Zb
    # (m, oh, ow, fh, fw, u)    - np.sum(Zb, axis=(5))
    # (m, oh, mw, fh, u)        - np.sum(Zb, axis=(5, 4))
    # (m, oh, ow, u)            - np.sum(Zb, axis=(5, 4, 3))

    # (5.3) Add bias to linear activation product
    layer.fc['Z'] += layer.p['b']

    # (6) Non-linear activation
    layer.fc['A'] = layer.activate(layer.fc['Z'])

    return layer.fc['A']    # To next layer

def optimized_convolution_forward(layer, A):
    """Forward propagate signal to next layer."""
    # (1) Initialize cache and pad image
    X = initialize_forward(layer, A)  # (m, h, w, d)

    # (2) Create empty output Z based on output dimensions
    m = X.shape[0]
    Z = np.zeros((m, layer.d['oh'], layer.d['ow'], layer.d['u']))

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