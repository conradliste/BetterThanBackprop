from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from betterthanbackprop.utils.torch_utils import NumpyLoader, RotatingSequence

def load_rotating_mnist(data_path, num_rots, train=True, download=False):
    """
    Loads a rotating mnist
    """
    # Load in the MNIST dataset
    mnist = MNIST(
                data_path,  
                train=train,  
                download=download, 
                transform=RotatingSequence(num_rots))
    return mnist


