

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def linear_forward(x, w, b):
    """Compute the forward pass for a linear (fully-connected) layer."""
    # weights act as transformation matrix.
    return jnp.dot(w, x) + b

def relu(x):
    """Compute the forward pass for a ReLU activation function."""
    return jnp.maximum(0, x)

def logsoftmax(logits):
    """
    Compute log-softmax for logits.
    This is more numerically stable than softmax when computing log-likelihoods.
    JAX's logsumexp function is numerically stable and avoids underflow/overflow.
    """
    return logits - logsumexp(logits)

def categorical_cross_entropy(y_true, y_pred):
    """Compute the cross-entropy loss between true and predicted labels."""
    # Assumes y_pred is in log probabilities, and y_true is one-hot encoded.
    return -jnp.mean(y_true * y_pred)
