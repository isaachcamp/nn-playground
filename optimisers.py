
from typing import List, Dict
import jax.numpy as jnp

class Optimiser:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def update_params(self, params: List[Dict], grads: List[Dict], epoch) -> List[Dict]:
        """Update parameters using the computed gradients."""
        raise NotImplementedError
    
class SGD(Optimiser):
    def __init__(self, learning_rate: float) -> None:
        super().__init__(learning_rate)

    def update_params(self, params: List[Dict], grads: List[Dict], epoch: int) -> List[Dict]:
        """Update parameters using the computed gradients."""
        lr = exponential_decay(self.learning_rate, epoch=epoch, decay_rate=0.1)

        for p, g in zip(params, grads):
            p['weights'] -= lr * g['weights']
            p['biases'] -= lr * g['biases']
        return params


def exponential_decay(lr: float, epoch: int = 1, decay_rate: float = 0.1) -> float:
    """Compute the learning rate decay using exponential decay."""
    return lr * jnp.exp(-decay_rate * epoch)
