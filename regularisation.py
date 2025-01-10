
import numpy as np

class Dropout:
    def __init__(self, p: float):
        """Initialize Dropout layer."""
        self.p = p

    def __call__(self, x):
        """Set some nodes to zero with some probability `p` during forward pass."""
        mask = np.random.rand(*x.shape) > self.p
        scaled_mask = mask / (1 - self.p) # Scale values to maintain expected value
        return x * scaled_mask
