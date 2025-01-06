
from abc import ABC, abstractmethod
import jax.numpy as jnp

from functional import linear_forward
from initialisations import initialise_weights


class Layer(ABC):
    def __init__(self, dims, activation) -> None:
        self.dims = dims
        self.activation = activation

    @abstractmethod
    def init_params(self):
        """Initialize parameters for the layer."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x, w, b):
        """Forward pass for the layer."""
        raise NotImplementedError


class Dense(Layer):
    """Implement fully-connected layer."""
    def init_params(self):
        weights = initialise_weights(self.dims, method='He')
        biases = jnp.zeros(self.dims[1])
        return {'weights': weights, 'biases': biases}

    def forward(self, x, w, b):
        return self.activation(linear_forward(x, w, b))


class Conv2D(Layer):
    """Implement 2D convolutional layer."""
    def __init__(self, dims, activation):
        super().__init__(dims, activation)

    def init_params(self):
        pass

    def forward(self, x, w, b):
        pass
