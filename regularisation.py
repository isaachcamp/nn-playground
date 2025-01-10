
import numpy as np
import jax.numpy as jnp

class Dropout:
    def __init__(self, p: float):
        """Initialize Dropout layer."""
        self.p = p

    def __call__(self, x):
        """Set some nodes to zero with some probability `p` during forward pass."""
        mask = np.random.rand(*x.shape) > self.p
        scaled_mask = mask / (1 - self.p) # Scale values to maintain expected value
        return x * scaled_mask

class L2Regularisation:
    def __init__(self, lambda_: float):
        """Initialize L2 Regularisation layer."""
        self.lambda_ = lambda_

    def __call__(self, params):
        """Compute L2 Regularisation loss."""
        sum_of_squares = [jnp.sum(jnp.square(layer['weights'])) for layer in params]
        return (self.lambda_ / 2) * sum(sum_of_squares)
