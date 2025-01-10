
from abc import ABC, abstractmethod
from typing import List, Dict
import jax.numpy as jnp

class Optimiser(ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.state = {'epoch': 0}

    def next_epoch(self):
        """Increment epoch counter."""
        self.state['epoch'] += 1

    def set_state(self, state: Dict) -> None:
        """Set state of optimiser to keep track of internal params."""
        # Required to remove side effects for JAX jit optimisation.
        self.state = state

    @abstractmethod
    def update_params(self, params: List[Dict], grads: List[Dict]) -> List[Dict]:
        """Update parameters using the computed gradients."""
        raise NotImplementedError


class SGD(Optimiser):
    """Stochastic Gradient Descent optimiser."""
    def update_params(self, params: List[Dict], grads: List[Dict]) -> List[Dict]:
        """Update parameters using pre-computed gradients."""
        epoch = self.state.get('epoch')
        lr = exponential_decay(self.learning_rate, epoch=epoch, decay_rate=0.1)

        for p, g in zip(params, grads):
            p['weights'] -= lr * g['weights']
            p['biases'] -= lr * g['biases']
        return params, self.state


class Momentum(Optimiser):
    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum

    def update_params(self, params: List[Dict], grads: List[Dict]) -> List[Dict]:
        """Update parameters using the computed gradients."""
        epoch = self.state.get('epoch')
        velocities = self.state.get('velocities')

        lr = exponential_decay(self.learning_rate, epoch=epoch, decay_rate=0.1)

        if velocities is None:
            velocities = [{} for _ in range(len(params))]
            for v, g in zip(velocities, grads):
                v['weights'] = jnp.zeros_like(g['weights'])
                v['biases'] = jnp.zeros_like(g['biases'])

        for p, v, g in zip(params, velocities, grads):
            v['weights'] = self.momentum * v['weights'] - lr * g['weights']
            v['biases'] = self.momentum * v['biases'] - lr * g['biases']
            p['weights'] += v['weights']
            p['biases'] += v['biases']

        return params, {**self.state, 'velocities': velocities}


class Adam(Optimiser):
    def __init__(
            self,
            learning_rate: float,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8
        ) -> None:

        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state['time'] = 0

    def update_params(self, params: List[Dict], grads: List[Dict]) -> List[Dict]:
        """Update parameters using the computed gradients."""
        lr = self.learning_rate
        t = self.state.get('time') + 1
        moment1 = self.state.get('moment1')
        moment2 = self.state.get('moment2')

        # Initialise moment1 and moment2 if not present in state.
        if moment1 is None:
            moment1 = [{} for _ in range(len(params))]
            moment2 = [{} for _ in range(len(params))]
            for m1, m2, g in zip(moment1, moment2, grads):
                m1['weights'] = jnp.zeros_like(g['weights'])
                m1['biases'] = jnp.zeros_like(g['biases'])
                m2['weights'] = jnp.zeros_like(g['weights'])
                m2['biases'] = jnp.zeros_like(g['biases'])

        for p, m1, m2, g in zip(params, moment1, moment2, grads):
            # Calculate 1st and 2nd moment estimates.
            m1['weights'] = self.beta1 * m1['weights'] + (1 - self.beta1) * g['weights']
            m1['biases'] = self.beta1 * m1['biases'] + (1 - self.beta1) * g['biases']

            m2['weights'] = self.beta2 * m2['weights'] + (1 - self.beta2) * (g['weights'] ** 2)
            m2['biases'] = self.beta2 * m2['biases'] + (1 - self.beta2) * (g['biases'] ** 2)

            # Bias correct moments.
            m1['weights'] /= (1 - self.beta1 ** t)
            m1['biases'] /= (1 - self.beta1 ** t)

            m2['weights'] /= (1 - self.beta2 ** t)
            m2['biases'] /= (1 - self.beta2 ** t)

            # Update parameters.
            p['weights'] -= lr * m1['weights'] / (jnp.sqrt(m2['weights']) + self.epsilon)
            p['biases'] -= lr * m1['biases'] / (jnp.sqrt(m2['biases']) + self.epsilon)

        # Return updated state variables explicitly.
        return params, {**self.state, 'time': t, 'moment1': moment1, 'moment2': moment2}


def exponential_decay(lr: float, epoch: int = 1, decay_rate: float = 0.1) -> float:
    """Compute the learning rate decay using exponential decay."""
    return lr * jnp.exp(-decay_rate * epoch)
