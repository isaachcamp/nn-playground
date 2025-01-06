
import jax
import jax.numpy as jnp

Array = jnp.ndarray

def random_init_weights(dims: tuple, key: Array) -> Array:
    """Generate random weights from Normal distribution and scale by 0.01."""
    return jax.random.normal(key, dims) * 0.01

def he_init_weights(dims: tuple, key: Array) -> Array:
    """Initialise weights using He (2015) method, with normalised variance."""
    return jax.random.normal(key, dims) * jnp.sqrt(2 / dims[0])

def initialise_weights(
        dims: tuple,
        method: str = 'default',
        seed: int = 42
    ) -> Array:
    """
    Initialise weights for a given layer.

    Parameters:
    method (str): The method to use for weight initialization. Options are 
        "dflt" for default random initialization and "He" for He initialization.
    dims (tuple): The dimensions of the weights to be initialized.
    seed (int, optional): The seed for the random number generator. Defaults 
        to None.

    Returns:
    np.ndarray: The initialized weights as a NumPy array.
    """

    key = jax.random.key(seed)

    if method == 'default':
        return random_init_weights(dims, key)
    elif method == 'He':
        return he_init_weights(dims, key)

    raise KeyError(f'Method "{method}" not recognised.')
