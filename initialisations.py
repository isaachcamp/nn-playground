
import numpy as np

def random_init_weights(dims: tuple) -> np.ndarray:
    """Generate random weights from Normal distribution and scale by 0.01."""
    return np.random.randn(*dims) * 0.01

def he_init_weights(dims: tuple) -> np.ndarray:
    """Initialise weights using He (2015) method, with normalised variance."""
    return np.random.randn(*dims) * np.sqrt(2 / dims[0])

def initialise_weights(
        dims: tuple,
        method: str = 'default',
        seed: int = None
    ) -> np.ndarray:
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

    if seed is not None:
        np.random.seed(seed)

    if method == 'default':
        return random_init_weights(dims)
    elif method == 'He':
        return he_init_weights(dims)

    raise KeyError(f'Method "{method}" not recognised.')