from torchvision.datasets import MNIST
from utils.torch_utils import NumpyLoader

def load_mnist(data_path, train_batch_size, test_batch_size, transform=None, download=False):
    train_dataset = MNIST(
                        data_path, 
                        train=True, 
                        download=download,
                        transform=transform)
    train_loader = NumpyLoader(
                            train_dataset,
                            batch_size=train_batch_size,
                            shuffle=True)
    test_dataset = MNIST(
                        data_path, 
                        train=False, 
                        download=download,
                        transform=transform)
    test_loader = NumpyLoader(
                        test_dataset, 
                        batch_size=test_batch_size,
                        shuffle=True)
    return train_dataset, test_dataset, train_loader, test_loader