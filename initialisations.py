
from jax import random
import jax.numpy as jnp

Array = jnp.ndarray

def random_init_weights(dims: tuple, key: Array, scale: float = 1e-2) -> Array:
    """Generate random weights from Normal distribution and scale."""
     # initialise as W^transpose to remove need for later transpositions.
    return scale * random.normal(key, (dims[-1], *dims[:-1]))

def he_init_weights(dims: tuple, key: Array) -> Array:
    """Initialise weights using He (2015) method, with normalised variance."""
    # Normalisation depends on size of previous layer
    prev_layer_size = jnp.array(dims[:-1]).prod()
     # initialise as W^transpose to remove need for later transpositions.
    return random.normal(key, (dims[-1], *dims[:-1])) * jnp.sqrt(2 / prev_layer_size)
