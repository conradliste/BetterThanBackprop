import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import diffrax
from diffrax import diffeqsolve, ODETerm, Tsit5

from betterthanbackprop.dataLoaders import rotating_mnist, mnist
from betterthanbackprop.utils import plot_utils, nn_utils, torch_utils
from betterthanbackprop.models.common_modules import MLP

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple MLP MNIST classifier')
    parser.add_argument('--config', help='Path to config file', required=True)
    args = parser.parse_args()

    # Parse config file to obtain hyperparameters
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        # Training hyperparameters
        train_hyperparams = config['training_hyperparams']
        data_path = train_hyperparams['data_path']
        n_epochs = train_hyperparams['n_epochs']
        train_batch_size = train_hyperparams['batch_size_train']
        test_batch_size = train_hyperparams['batch_size_test']
        # Optimizer hyperparameters
        optim_hyperparams = config['optim_hyperparams']
        learning_rate = optim_hyperparams['learning_rate']
        beta1 = optim_hyperparams['beta1']
        beta2 = optim_hyperparams['beta2']
        eps = optim_hyperparams['eps']
        # Logging hyperparameters
        log_hyperparams = config['logging_params']
        log_interval = log_hyperparams['log_interval']
        # Model hyperparameters
        model_hyperparams = config['model_hyperparams']
        hidden_dims = model_hyperparams['hidden_dims']
        activations = model_hyperparams['activations']
        output_dim = model_hyperparams['output_dim']

# Encoder


# Decoder

# NeuralODE
class FirstOrderDynamics(nn.Module):
    """
    Neural Net modeling the first derivative of the function
    """
    @nn.compact
    def __call__(self, x):
        """
        Forward pass
        """
        x = nn.Conv(features=32, kernel_size=(5,5))(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(5,5))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(5,5))(x)
        x = nn.Dense(256)(x)
        x = nn.Dense(10)(x)                              

class NeuralODE:
    def __init__(self, net, solver):
        self.net = net
        self.solver = solver
        self.times = 
        return
    
    def func(self, t, y, args):
        """
        This function just formats the neural network for diffrax
        which requires the input to have t, y, and args as arguments
        """
        return self.net.apply({'params': args}, y)
    
    def __call__(self, params, y0):
        solution = diffeqsolve(
            ODETerm(self.func),
            Tsit5(),
            t0=0,
            t1=1,
            dt0=None,
            y0=y0,
            args=params,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=[1]),
        )
        return solution.ys[0]

# Load Dataset
train_dataset, test_dataset, train_loader, test_loader = mnist.load_mnist(data_path, train_batch_size, test_batch_size, transform=torch_utils.JnpCast())
firstDeriv = FirstOrderDynamics()
neuralODE = NeuralODE(firstDeriv, Tsit5())
# Initialize the parameters
seed = jax.random.PRNGKey(0)
seed, init_seed = jax.random.split(seed)
params = nn_utils.init_params(firstDeriv, (1, 28, 28), init_seed)["params"]
print(jax.tree_map(lambda x: x.shape, params))
# Instantiate the optimizer
optimizer = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2, eps=eps)
opt_state = optimizer.init(params)
# Define the train step
@jax.jit
def train_step(params, inputs, labels, opt_state):
    # Loss function
    def loss_fn(params):
        logits = neuralODE(params, inputs)
        return nn_utils.cross_entropy_loss(logits, labels), logits
    # Evaluate loss and find gradient
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(params)
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss
# Define the eval step
@jax.jit
def eval_step(params, inputs, labels):
    # Forward pass
    logits = neuralODE(params, inputs)
    # Evaluate loss and accuracy
    loss = nn_utils.cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy
# Training Loop
# for epoch in range(n_epochs):
#     losses = []
#     # Train for one epoch
#     for i, (inputs, labels) in enumerate(train_loader):
#         params, opt_state, loss = train_step(params, inputs, labels, opt_state)
#         losses.append(loss)
#     print("Epoch: {}, Train Loss: {}".format(epoch, sum(losses)/len(losses)))
#     # Evaluate model
#     losses = []
#     accuracies = []
#     for i, (inputs, labels) in enumerate(test_loader):
#         loss, accuracy = eval_step(params, inputs, labels)
#         losses.append(loss)
#         accuracies.append(accuracy)
#     print("Epoch: {}, Test Loss: {}, Test Accuracy: {}".format(epoch, sum(losses)/len(losses), sum(accuracies)/len(accuracies)))