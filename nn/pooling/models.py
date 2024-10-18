# nn/nn/pool/models.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nn.commons.models import Layer
from nn.pooling.forward import pooling_forward, optimized_pooling_forward
from nn.pooling.backward import pooling_backward, optimized_pooling_backward
from nn.pooling.parameters import (
    pooling_compute_shapes,
    pooling_initialize_parameters,
    pooling_compute_gradients,
    pooling_update_parameters
)


class Pooling(Layer):
    """
    Definition of a pooling layer prototype.

    :param pool_size: Height and width for pooling window, defaults to `(2, 2)`.
    :type pool_size: int or tuple[int], optional

    :param strides: Height and width to shift the pooling window by, defaults to `None` which equals `pool_size`.
    :type strides: int or tuple[int], optional

    :param pool: Pooling activation of units, defaults to :func:`np.max`. Use one of min or max pooling.
    :type pool: function, optional
    """

    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 pool=np.max,
                 optimize = False):

        super().__init__()

        pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        strides = strides if isinstance(strides, tuple) else pool_size

        self.d['ph'], self.d['pw'] = pool_size
        self.d['sh'], self.d['sw'] = strides
        self.pool = pool

        self.activation = { 'pool': self.pool.__name__ }
        self.trainable = False
        self.optimize = optimize
        
        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nn.pooling.parameters.pooling_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        pooling_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`nn.pooling.parameters.initialize_parameters()`.
        """
        pooling_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`nn.pooling.forward.pooling_forward()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        self.compute_shapes(A)
        if (self.optimize):
            A = optimized_pooling_forward(self, A)
        else:
            A = pooling_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`nn.pooling.backward.pooling_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        if (self.optimize):
            dX = optimized_pooling_backward(self, dX)
        else:
            dX = pooling_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`nn.pooling.parameters.pooling_compute_gradients()`. Dummy method, there are no gradients to compute in layer.
        """
        pooling_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nn.pooling.parameters.pooling_update_parameters()`. Dummy method, there are no parameters to update in layer.
        """
        if self.trainable:
            pooling_update_parameters(self)

        return None

    def init(self):
        self.initialize_parameters()