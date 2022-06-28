import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np

from betterthanbackprop.dataLoaders import rotating_mnist
from betterthanbackprop.utils import plot_utils


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
        


