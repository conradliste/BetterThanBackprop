from typing import (Any, Callable, Sequence)
from typing import Any 

# Jax imports
import flax.linen as nn

class MLP(nn.Module):
    """
    Class to create an MLP network with arbitrary layer dims and activations
    """
    hidden_dims: Sequence[int]
    act: Callable[..., Any] = nn.relu
    use_bias: bool = True
    kernel_init: Callable[..., Any] = nn.initializers.lecun_normal()
    bias_init: Callable[..., Any] = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through MLP network
        """
        for dim in self.hidden_dims[:-1]:
            x = nn.Dense(dim, use_bias=self.use_bias, kernel_init=self.kernel_init,
                        bias_init=self.bias_init)(x)
            x = self.act(x)
        x = nn.Dense(self.hidden_dims[-1], use_bias=self.use_bias, 
                    kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x

class ConvNet(nn.Module):
    """
    Class to create a convolutional network with arbitrary layer dims and activations
    """
    hidden_dims: Sequence[int]
    activations: str = nn.relu

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through MLP network
        """
        for dim, act in zip(self.hidden_dims, self.activations):
            x = act(nn.Conv(dim)(x))
        return x
    
                
    
