import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import get_optimizer


def get_param_matrix(model_prefix, model_dir):
    """
    Grabs the parameters of a saved model and returns them as a matrix


    """
    # Load and combine the parameters
    param_matrix = []
    for file in os.listdir(model_dir):
        if file.startswith(model_prefix):
            model_path = os.path.join(model_dir, file)
            state_dict = torch.load(model_path)
            # Grab all params in state dict
            params = [state_dict[param].data.float() for param in state_dict]  
            # Reshape to one long parameter vector
            params = nn.utils.parameters_to_vector(params)
            param_matrix.append(params.cpu().numpy())
    params_matrix = np.array(param_matrix)
    return params_matrix

def plot_trajectory(projected_params):
    # Separate components
    x = projected_params[:, 0]
    y = projected_params[:, 1]
    z = projected_params[:, 2]
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    # Creating plot
    ax.scatter3D(x, y, z, color="green")
    plt.title("Projected Learning Trajectory")
