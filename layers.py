
from abc import ABC, abstractmethod
import jax.numpy as jnp

from functional import linear_forward
from initialisations import random_init_weights


class Layer(ABC):
    def __init__(self, dims) -> None:
        self.dims = dims

    @abstractmethod
    def init_params(self, key):
        """Initialize parameters for the layer."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x, w, b):
        """Forward pass for the layer."""
        raise NotImplementedError


class Dense(Layer):
    """Implement fully-connected layer."""
    def init_params(self, key):
        w = random_init_weights(self.dims, key)
        b = jnp.zeros((self.dims[-1],))
        return {'weights': w, 'biases': b}

    def forward(self, x, w, b):
        return linear_forward(x, w, b)


class Conv2D(Layer):
    """Implement 2D convolutional layer."""
    def __init__(self, dims):
        super().__init__(dims)

    def init_params(self):
        pass

    def forward(self, x, w, b):
        pass
