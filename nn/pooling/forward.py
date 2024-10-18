# nn/nn/pooling/forward.py
# Related third party imports
import numpy as np

def pooling_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    layer.fc['X'] = A

    # (2) Slice input w.r.t. pool size (ph, pw) and strides (sh, sw)
    layer.fc['Xb'] = np.array([[layer.fc['X'][ :, h:h + layer.d['ph'], w:w + layer.d['pw'], :]
                    # Inner loop
                    # (m, h, w, d) ->
                    # (ow, m, h, pw, d)
                    for w in range(layer.d['w'] - layer.d['pw'] + 1)
                    if w % layer.d['sw'] == 0]
                # Outer loop
                # (ow, m, h, pw, d) ->
                # (oh, ow, m, ph, pw, d)
                for h in range(layer.d['h'] - layer.d['ph'] + 1)
                if h % layer.d['sh'] == 0])

    # (3) Bring back m along axis 0
    layer.fc['Xb'] = np.moveaxis(layer.fc['Xb'], 2, 0)
    # (oh, ow, m, ph, pw, d) ->
    # (m, oh, ow, ph, pw, d)

    # (4) Apply pooling operation on blocks
    layer.fc['Z'] = layer.pool(layer.fc['Xb'], axis=(4, 3))
    # (m, oh, ow, ph, pw, d) - Xb
    # (m, oh, ow, ph, d)     - layer.pool(Xb, axis=4)
    # (m, oh, ow, d)         - layer.pool(Xb, axis=(4, 3))

    A = layer.fc['A'] = layer.fc['Z']

    return A    # To next layer

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
