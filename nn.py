
from jax import value_and_grad
import jax.numpy as jnp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from layers import Dense
from functional import relu_forward, softmax, categorical_cross_entropy_loss
from optimisers import exponential_decay
from manipulate_data import one_hot, flatten


LEARNING_RATE = 0.01
EPOCHS = 50


class NeuralNetwork:
    def __init__(self) -> None:
        self.layers = [
            Dense((784, 128), activation=relu_forward),
            Dense((128, 24), activation=relu_forward),
            Dense((24, 10), activation=softmax)
        ]

    def init_params(self):
        return [layer.init_params() for layer in self.layers]

    def forward(self, params, x):
        for layer, p in zip(self.layers, params):
            x = layer.forward(x, p['weights'], p['biases'])
        return x

    def __call__(self, params, x):
        return self.forward(params, x)

    @staticmethod
    def update(params, grads, lr):
        for p, g in zip(params, grads):
            p['weights'] -= lr * g['weights']
            p['biases'] -= lr * g['biases']
        return params


def loss(params, model, x, y_true):
    y_pred = model(params, x)
    return categorical_cross_entropy_loss(y_true, y_pred)


if __name__ == '__main__':
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

    nn = NeuralNetwork()
    params = nn.init_params()
    costs = []

    for epoch in range(EPOCHS):
        for train_data, train_labels in iter(train_dataloader):
            inputs = flatten(jnp.array(train_data.numpy()))
            targets = jnp.array(one_hot(train_labels.numpy(), num_classes=10))

            cost, grads = value_and_grad(loss)(params, nn, inputs, targets)
            costs.append(cost)

            learning_rate = exponential_decay(LEARNING_RATE, step=epoch)
            params = nn.update(params, grads, learning_rate)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {cost}')
