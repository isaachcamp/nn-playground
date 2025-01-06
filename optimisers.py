
import jax.numpy as jnp

def exponential_decay(learning_rate: float, decay_rate: float = 0.1, step: int = 1) -> float:
    """Compute the learning rate decay using exponential decay."""
    return learning_rate * jnp.exp(-decay_rate * step)

