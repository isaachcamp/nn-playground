
from functools import partial
import time

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from layers import Dense
from manipulate_data import one_hot, flatten
from metrics import accuracy
import functional as F
from optimisers import Optimiser, Adam
from regularisation import Dropout


EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
DROPOUT = 0.05


class FlattenAndCast(object):
    """Data transform to flatten images and cast as type JAX float32."""
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))

class NeuralNetwork:
    def __init__(
            self,
            layers,
            params,
            optimiser: Optimiser,
            dropout: Dropout = Dropout(DROPOUT),
            training: bool = True
        ) -> None:

        self.layers = layers
        self.set_params(params)
        self.optimiser = optimiser
        self.dropout = dropout
        self.training = training

    def set_params(self, params):
        self.params = params
    
    def predict(self, params, x):
        activations = x
        for p in params[:-1]:
            outputs = F.linear_forward(activations, p['weights'], p['biases'])
            outputs = self.dropout(outputs) if self.training else outputs
            activations = F.relu(outputs)

        final_p = params[-1]
        logits = F.linear_forward(activations, final_p['weights'], final_p['biases'])
        return F.logsoftmax(logits)

    def __call__(self, params, x):
        return self.predict(params, x)

# Lambda fn, Batched version of `nn.predict`
batched_predict = vmap(NeuralNetwork.__call__, in_axes=(None, None, 0))

def loss(params, model, inputs, targets):
    y_pred = batched_predict(model, params, inputs) # Batched version of `nn.predict`
    return F.categorical_cross_entropy(targets, y_pred)

@partial(jit, static_argnames=['model'])
def update(params, x, y, model):
    grads = grad(loss)(params, model, x, y)
    return model.optimiser.update_params(params, grads)

def init_params(layers, key):
    """Initialise parameters for all layers in the network."""
    # Collect parameters for JAX grad computation.
    keys = random.split(key, len(layers))
    return [layer.init_params(key) for key,layer in zip(keys, layers)]

if __name__ == '__main__':
    # Set up neural network.
    n_classes = 10
    input_size = 28 * 28

    layers = [
            Dense((input_size, 512)),
            Dense((512, 512)),
            Dense((512, n_classes)),
        ]
    params = init_params(layers, random.key(0))
    optimiser = Adam(LEARNING_RATE)
    nn = NeuralNetwork(layers, params, optimiser)

    # Load data and create batched generator `dataloader`.
    train_dataset = MNIST('./data', train=True, download=True, transform=FlattenAndCast())
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get full test and train datasets (for checking accuracy)
    train_images = flatten(jnp.array(train_dataset.train_data))
    train_labels = one_hot(train_dataset.train_labels.numpy(), n_classes)

    test_dataset = MNIST('./data', train=False, download=True)
    test_images = flatten(jnp.array(test_dataset.test_data.numpy()))
    test_labels = one_hot(test_dataset.test_labels.numpy(), n_classes)

    # Training loop
    train_accs = []
    test_accs = []
    costs = []

    for epoch in range(EPOCHS):
        start_time = time.time()
        for x, y in train_dataloader:
            inputs = flatten(jnp.array(x.numpy()))
            targets = one_hot(y.numpy(), n_classes)

            # Backpropagate gradients and update parameters.
            params, optimiser_state = update(params, inputs, targets, nn)
            nn.set_params(params)
            nn.optimiser.set_state(optimiser_state) # Removes side effects from `update`.

        # Evaluation
        epoch_time = time.time() - start_time

        nn.training = False # Disable dropout for evaluation
        train_acc = accuracy(train_labels, y_pred=batched_predict(nn, params, train_images))
        test_acc = accuracy(test_labels, y_pred=batched_predict(nn, params, test_images))
        cost = loss(params, nn, train_images, train_labels)
        nn.training = True

        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        costs.append(cost)

        optimiser.next_epoch()

    # Accuracy plot
    x = np.arange(len(train_accs))

    plt.plot(x, train_accs, label='Train')
    plt.plot(x, test_accs, label='Test', linestyle='--')
    ax = plt.gca()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy Cost')
    ax.legend()
    plt.savefig('test_train_accuracy_nn.png')
    plt.close()

    # Loss plot
    plt.plot(np.arange(len(costs)), costs)
    ax = plt.gca()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy Cost')
    plt.savefig('cost_nn.png')
    plt.close()
