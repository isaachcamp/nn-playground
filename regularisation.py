
import numpy as np

class Dropout:
    def __init__(self, drop_prob: float):
        """Initialize Dropout layer."""
        self.drop_prob = drop_prob

    def __call__(self, x):
        """Apply dropout during forward pass."""
        # Mask input values with probability drop_prob
        mask = np.random.rand(*x.shape) > self.drop_prob
        scaled_mask = mask / (1 - self.drop_prob) # Scale values to maintain expected value
        return x * scaled_mask
