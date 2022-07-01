import argparse
import numpy as np
import matplotlib.pyplot as plt

from betterthanbackprop.utils import plot_utils
from betterthanbackprop.dataLoaders import rotating_mnist


def test_rotation(data_path):
    """
    Displays all the rotations of a random MNIST digit
    """
    train_set, test_set, train_dataloader, test_dataloader = rotating_mnist.load_rotating_mnist(data_path, 10, 32, 32, train=True)
    random_index = np.random.randint(0, len(train_set))
    rand_image = train_set[random_index][0]
    print("Data Type:", type(rand_image))
    print("Shape:", rand_image.shape)
    plot_utils.display_images(rand_image)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests for Rotating Mnist')
    parser.add_argument('--data-path', help='the path to data', required=True)
    args = parser.parse_args()
    test_rotation(args.data_path)
    