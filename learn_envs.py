import gym
from gym import Env
import numpy as np
import torch

class LearnEnv(Env):
    def __init_(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.num_params = sum(p.numel() for p in self.model.parameters()) 
        self.state = torch.zeros(num_params + 4)
        return
    
    def step(self, action):
        # Flatten images
        images = images.view(-1, 28*28)
        # Forward pass
        output = self.model(images)
        # Compute loss
        loss = self.loss(output, labels)
        # Clear gradient buffer   
        self.optimizer.zero_grad()         
        # Backpropagate to get gradients 
        loss.backward()    
        # Step in the direction of negative gradient
        self.optimizer.step()
        print ('Epoch [{}/{}], Step {}, Loss: {:.4f}' 
                .format(epoch + 1, num_epochs, batch_idx + 1, loss.item()))
        return
    
    def reset(self):
        return
    