
import jax.numpy as jnp

def accuracy(targets, y_pred):
    """Compute the accuracy of a model's predictions."""
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(y_pred, axis=1)
    return jnp.mean(predicted_class == target_class)
