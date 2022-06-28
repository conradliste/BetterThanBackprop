
import torch.optim as optim
import numpy as np
import scipy.ndimage
import jax.numpy as jnp
#
from torch.utils import data
import torch.nn as nn
import torch
import os


class FlattenAndCast(object):
    """
    Helper class that acts a transform to flatten and cast the input to a jnp.float32
    """
    def __call__(self, input):
        return jnp.array(input).reshape(28*28)

class JnpCast(object):
    def __call__(self, input):
        return jnp.array(input)

class RotatingSequence(object):
    """
    Generates a sequence an input image rotate a specfic 
    """
    def __init__(self, num_rots):
        self.num_rots = num_rots
        return

    def __call__(self, input):
        rot_sequence = []
        rot_angle = 360 / self.num_rots
        for rot_idx in range(self.num_rots + 1):
            rot_image = scipy.ndimage.rotate(input, rot_angle * rot_idx, reshape=False)
            rot_sequence.append(rot_image)
        return rot_sequence


class NumpyLoader(data.DataLoader):
    """
    Class that loads Torch datasets in based only the transforms specified 
    (the default PyTorch dataloader will convert batches to Torch tensors
    automatically)
    """
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

def numpy_collate(batch):
    """
    Collate Function for NumpyLoader above
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def get_torch_optimizer(optim_name, params, lr=0.001):
    """
    Returns optimizers by name

    Args:
        optim_name (str): name of the optimizer
        params (): parameters to optimize
    Returns:
        The optimizer
    """
    if optim_name == 'Adam':
        optimizer = optim.Adam(params, lr=lr)
    elif optim_name == 'NAdam':
        optimizer = optim.NAdam(params, lr=lr)
    elif optim_name == 'RAdam':
        optimizer = optim.RAdam(params, lr=lr)
    elif optim_name == 'AdamW':
        optimizer = optim.AdamW(params, lr=lr)
    elif optim_name == 'SparseAdam':
        optimizer = optim.SparseAdam(params, lr=lr)
    elif optim_name == 'SGD':
        optimizer = optim.SGD(params, lr=lr)
    elif optim_name == 'ASGD':
        optimizer = optim.ASGD(params, lr=lr)
    elif optim_name == 'RMSprop':
        optimizer = optim.RMSprop(params, lr=lr)
    elif optim_name == 'Rprop':
        optimizer = optim.Rprop(params, lr=lr)
    elif optim_name == 'Adadelta':
        optimizer = optim.Adadelta(params, lr=lr)
    elif optim_name == 'Adagrad':
        optimizer = optim.Adagrad(params, lr=lr)
    elif optim_name == 'Adamax':
        optimizer = optim.Adamax(params, lr=lr)
    elif optim_name == 'LBFGS':
        optimizer = optim.LBFGS(params, lr=lr)
    return optimizer

def print_state_dict(state_dict):
    """
    Prints the state dict of a model
    """
    for param_tensor in state_dict:
        print(param_tensor, "\t", state_dict[param_tensor].size())
    print("\n")
    return

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