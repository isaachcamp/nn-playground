
import jax.numpy as jnp

def one_hot(x, num_classes):
    """Create a one-hot encoding of x for n classes."""
    return jnp.array(x[:, None] == jnp.arange(num_classes), jnp.float32)

def flatten(x):
    """Flatten a multi-dimensional array x."""
    return x.reshape(x.shape[0], -1)
