
from typing import List, Dict
import jax.numpy as jnp

class Optimiser:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def update_params(self, params: List[Dict], grads: List[Dict]) -> List[Dict]:
        """Update parameters using the computed gradients."""
        raise NotImplementedError
    
class SGD(Optimiser):
    def __init__(self, learning_rate: float) -> None:
        super().__init__(learning_rate)

    def update_params(self, params: List[Dict], grads: List[Dict]) -> List[Dict]:
        """Update parameters using the computed gradients."""
        for p, g in zip(params, grads):
            p['weights'] -= self.learning_rate * g['weights']
            p['biases'] -= self.learning_rate * g['biases']
        return params

def exponential_decay(learning_rate: float, decay_rate: float = 0.1, step: int = 1) -> float:
    """Compute the learning rate decay using exponential decay."""
    return learning_rate * jnp.exp(-decay_rate * step)
