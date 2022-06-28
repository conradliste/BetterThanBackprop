import yaml
import argparse
import optax
import jax.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
# Custom modules
from betterthanbackprop.models.common_modules import MLP
from betterthanbackprop.dataLoaders import mnist
from betterthanbackprop.utils.nn_utils import cross_entropy_loss, init_params
from betterthanbackprop.utils.torch_utils import FlattenAndCast

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

    # Load in the data
    train_dataset, test_dataset, train_loader, test_loader = mnist.load_mnist(data_path, train_batch_size, test_batch_size, transform=FlattenAndCast())
    # Create the network
    model = MLP(hidden_dims=hidden_dims + [output_dim], act=nn.relu)
    # Initialize the parameters
    seed = jax.random.PRNGKey(0)
    seed, init_seed = jax.random.split(seed)
    params = init_params(model, (1, 784), init_seed)["params"]
    print(jax.tree_map(lambda x: x.shape, params))
    # Instantiate the optimizer
    optimizer = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2, eps=eps)
    opt_state = optimizer.init(params)
    # Define the train step
    @jax.jit
    def train_step(params, inputs, labels, opt_state):
        # Loss function
        def loss_fn(params):
            logits = model.apply({'params': params}, inputs)
            return cross_entropy_loss(logits, labels), logits
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
        logits = model.apply({'params': params}, inputs)
        # Evaluate loss and accuracy
        loss = cross_entropy_loss(logits=logits, labels=labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return loss, accuracy
    # Training Loop
    for epoch in range(n_epochs):
        losses = []
        # Train for one epoch
        for i, (inputs, labels) in enumerate(train_loader):
            params, opt_state, loss = train_step(params, inputs, labels, opt_state)
            losses.append(loss)
        print("Epoch: {}, Train Loss: {}".format(epoch, sum(losses)/len(losses)))
        # Evaluate model
        losses = []
        accuracies = []
        for i, (inputs, labels) in enumerate(test_loader):
            loss, accuracy = eval_step(params, inputs, labels)
            losses.append(loss)
            accuracies.append(accuracy)
        print("Epoch: {}, Test Loss: {}, Test Accuracy: {}".format(epoch, sum(losses)/len(losses), sum(accuracies)/len(accuracies)))