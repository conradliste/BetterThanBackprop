import argparse
import numpy as np
import matplotlib.pyplot as plt

from betterthanbackprop.utils import plot_utils
from betterthanbackprop.dataLoaders import rotating_mnist


def test_rotation(data_path):
    """
    Displays all the rotations of a random MNIST digit
    """
    train_set = rotating_mnist.load_rotating_mnist(data_path, 10, train=True)
    random_index = np.random.randint(0, len(train_set))
    plot_utils.display_images(train_set[random_index][0])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests for Rotating Mnist')
    parser.add_argument('--data-path', help='the path to data', required=True)
    args = parser.parse_args()
    test_rotation(args.data_path)
    