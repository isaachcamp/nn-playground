
from jax import value_and_grad
import jax.numpy as jnp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from layers import Dense
from regularisation import Dropout
from functional import relu_forward, softmax, categorical_cross_entropy_loss
from optimisers import exponential_decay
from manipulate_data import one_hot, flatten


LEARNING_RATE = 0.01
EPOCHS = 50


class NeuralNetwork:
    def __init__(self, layers, params, training: bool = True) -> None:
        self.layers = layers
        self.set_params(params)
        self.training = training

    def set_params(self, params):
        self.params = params

    def forward(self, params, x, dropout=Dropout(0.2)):
        for layer, p in zip(self.layers[:-1], params[:-1]):
            x = layer.forward(x, p['weights'], p['biases'])

            if isinstance(layer, Dense) and self.training:
                x = dropout(x)

        x = self.layers[-1].forward(x, params[-1]['weights'], params[-1]['biases'])
        return x

    def __call__(self, params, x):
        return self.forward(params, x)

    def update(self, params, grads, lr):
        for p, g in zip(params, grads):
            p['weights'] -= lr * g['weights']
            p['biases'] -= lr * g['biases']
        self.set_params(params)


def init_params(layers):
    """Initialise parameters for all layers in the network."""
    # Collect parameters for JAX grad computation.
    return [layer.init_params() for layer in layers]

def loss(params, model, x, y_true):
    y_pred = model(params, x)
    return categorical_cross_entropy_loss(y_true, y_pred)


if __name__ == '__main__':

    # Load MNIST dataset.
    train_dataset = datasets.MNIST('./data', train=True,   download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ]))

    test_dataset = datasets.MNIST('./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Initialize neural network parameters.
    # Set up neural network.
    layers = [
            Dense((784, 512), activation=relu_forward),
            Dense((512, 512), activation=relu_forward),
            Dense((512, 24), activation=relu_forward),
            Dense((24, 10), activation=softmax)
        ]
    params = init_params(layers)
    nn = NeuralNetwork(layers, params)

    # Train the neural network.
    costs = []
    for epoch in range(EPOCHS):
        for train_data, train_labels in iter(train_dataloader):
            inputs = flatten(jnp.array(train_data.numpy()))
            targets = jnp.array(one_hot(train_labels.numpy(), num_classes=10))

            cost, grads = value_and_grad(loss)(nn.params, nn, inputs, targets)

            learning_rate = exponential_decay(LEARNING_RATE, step=epoch)
            nn.update(nn.params, grads, learning_rate)

        costs.append(cost)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {cost}')
