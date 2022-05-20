# Optimizer
import torch.optim as optim

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