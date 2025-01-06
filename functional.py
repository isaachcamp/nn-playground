

import jax.numpy as jnp


def linear_forward(x, w, b):
    """Compute the forward pass for a linear (fully-connected) layer."""
    return jnp.dot(x, w) + b

def relu_forward(x):
    """Compute the forward pass for a ReLU activation function."""
    return jnp.maximum(0, x)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum()

def categorical_cross_entropy_loss(y_true, y_pred):
    """Compute the cross-entropy loss between true and predicted labels."""
    epsilon = 1e-8 # for numerical stability
    return -jnp.mean(y_true * jnp.log1p(y_pred + epsilon - 1))
