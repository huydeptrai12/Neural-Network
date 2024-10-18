# nn/nn/pool/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of pooling layer.
    :type layer: :class:`nn.pooling.models.Pooling`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    return dA


def pooling_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache
    layer.bc['dA'] = dX  # (m, oh, ow, d)

    # (2) Restore pooling block axes
    dZ = dX
    dZ = np.expand_dims(dZ, axis=3)
    dZ = np.expand_dims(dZ, axis=3)
    # (m, oh, ow, d)         ->
    # (m, oh, ow, 1, d)      ->
    # (m, oh, ow, 1, 1, d)

    # (3) Initialize backward output dL/dX
    dX = np.zeros_like(layer.fc['X'])      # (m, h, w, d)

    # Iterate over forward output height
    for oh in range(layer.d['oh']):

        hs = oh * layer.d['sh']
        he = hs + layer.d['ph']

        # Iterate over forward output width
        for ow in range(layer.d['ow']):

            ws = ow * layer.d['sw']
            we = ws + layer.d['pw']

            # (4hw) Retrieve input block
            Xb = layer.fc['Xb'][:, oh, ow, :, :, :]
            # (m, oh, ow, ph, pw, d)  - Xb (array of blocks)
            # (m, ph, pw, d)          - Xb (single block)

            # (5hw) Retrieve pooled value and restore block shape
            Zb = layer.fc['Z'][:, oh:oh+1, ow:ow+1, :]
            Zb = np.repeat(Zb, layer.d['ph'], axis=1)
            Zb = np.repeat(Zb, layer.d['pw'], axis=2)
            # (m, oh, ow, d)    - Z
            # (m,  1,  1, d)    - Zb -> np.repeat(Zb, pw, axis=1)
            # (m, ph,  1, d)         -> np.repeat(Zb, pw, axis=2)
            # (m, ph, pw, d)

            # (6hw) Match pooled value in Zb against Xb
            mask = (Zb == Xb)

            # (7hw) Retrieve gradient w.r.t Z and restore block shape
            dZb = dZ[:, oh, ow, :]
            dZb = np.repeat(dZb, layer.d['ph'], 1)
            dZb = np.repeat(dZb, layer.d['pw'], 2)
            # (m, oh, ow,  1,  1, d) - dZ
            #         (m,  1,  1, d) - dZb -> np.repeat(dZb, ph, axis=1)
            #         (m, ph,  1, d)       -> np.repeat(dZb, pw, axis=2)
            #         (m, ph, pw, d)

            # (8hw) Keep dXb values for coordinates where Zb = Xb (mask)
            dXb = dZb * mask

            # (9hw) Gradient of the loss w.r.t Xb
            dX[:, hs:he, ws:we, :] += dXb
            # (m, ph, pw, d) - dX[:, hs:he, ws:we, :]
            # (m, ph, pw, d) - dXb

    layer.bc['dX'] = dX

    return dX

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
