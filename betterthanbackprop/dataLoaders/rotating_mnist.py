from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from betterthanbackprop.utils.torch_utils import NumpyLoader, RotatingSequence

def load_rotating_mnist(data_path, num_rots, train_batch_size, test_batch_size, train=True, download=False):
    """
    Loads a rotating mnist
    """
    train_dataset = MNIST(
                data_path,  
                train=True,  
                download=download, 
                transform=RotatingSequence(num_rots))
    train_loader = NumpyLoader(
                        train_dataset,
                        batch_size=train_batch_size,
                        shuffle=True)
    test_dataset = MNIST(
                data_path,  
                train=False,  
                download=download, 
                transform=RotatingSequence(num_rots))
    test_loader = NumpyLoader(
                        test_dataset, 
                        batch_size=test_batch_size,
                        shuffle=True)    
    return train_dataset, test_dataset, train_loader, test_loader


