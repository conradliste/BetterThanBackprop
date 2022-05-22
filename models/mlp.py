import torch

# Jax imports
import jax.numpy as jnp
from jax import grad, jit, vmap
import flax.linen as nn

class MLP(nn.Module):
    """
    Class to create an MLP network with arbitrary layer dims and activations
    """
    def setup(self, hidden_dims, activations=None):
        """
        Sets up architecture of MLP network

        Args:
            hidden_dims (Sequence[int]): The dimensions of the hidden layers
            activations (Sequence[str]): The activation functions for each layer
        """
        # Default use ReLU activations
        if activations is None:
            self.activations = [nn.relu] * len(hidden_dims)
        else:
            self.activations = activations
        # Create layer with correct hidden dims
        self.layers = []
        for dim in hidden_dims:
            self.layers.append(nn.Dense(dim))
        return

    def __call__(self, x):
        """
        Forward pass through MLP network
        """
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x
    
                
    
