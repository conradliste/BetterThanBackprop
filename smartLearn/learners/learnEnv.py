from gym import Env
import numpy as np
import torch
import jax.numpy as jnp
import optax

class LearnEnv(Env):
    def __init_(self, model, loss, cost, dataloader, optimizer, seed=0):
        """
        Contructor for learning environment
        
        Args:
            model (Callable): Model to be trained
            loss (Callable): Loss function
            cost (Callable): Cost function
            dataloader(torch.utils.data.DataLoader): Data loader
            grad_controller(GradController): Takes in parameters
        """
        # Store arguments
        self.model = model
        self.loss = loss
        self.cost = cost
        self.dataloader = dataloader
        # Default to Adam optimizer
        if optimizer is None:
            self.optimizer = optax.Adam(model.parameters())
        else:
            self.optimizer = optimizer
    
    def step(self, control):
        """
        Advances the learning state by one step
        """
        return

    def reset(self):
        """
        Resets the learning state and reinitializes the model parameters
        """
        return
    