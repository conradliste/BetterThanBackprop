
# Imports
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn as nn
import torch
import torch.functional as F
import sys
import os
import matplotlib.pyplot as plt

# Set Hyperparameters
n_epochs = 10
batch_size_train = 50
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# Transformation to preprocess the data
transform = transforms.Compose([
    transforms.ToTensor()
])
# Load the data
data_path= ""
train_dataset = MNIST(data_path, train=True, download=True,
    transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_dataset = MNIST(data_path, train=False, download=True,
    transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size_test, shuffle=True)

# View the data
for batch_idx, (image, label) in enumerate(train_loader):
    print("The dimensions of the image are:", image.shape)
    print("The label of the first image is:", label[0])
    plt.imshow(image[0].numpy().squeeze(), cmap='gray')
    plt.show()
    break

# Create the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mlp1 = nn.Linear(784, 64)
        self.mlp2 = nn.Linear(64, 32)
        self.mlp3 = nn.Linear(32, 10)
        self.mlp4 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x
network = Net()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

# Train Loop
def train(num_epochs, network, train_loader):
    # Set network to train mode (ensures parameters are updated)
    network.train()
    for epoch in range(num_epochs):
        # Loop over the whole training set once
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Flatten images
            images = images.view(-1, 28*28)
            # Forward pass
            output = network(images)
            # Compute loss
            loss = loss_func(output, labels)
            # Clear gradient buffer   
            optimizer.zero_grad()         
            # Backpropagate to get gradients 
            loss.backward()    
            # Step in the direction of negative gradient
            optimizer.step()
            print ('Epoch [{}/{}], Step {}, Loss: {:.4f}' 
                    .format(epoch + 1, num_epochs, batch_idx + 1, loss.item()))
train(n_epochs, network, train_loader)


# Test Loop
def test(network, test_loader):
    # Ensures parameters are not updated
    network.eval()
    # Ensures no gradients are computed
    with torch.no_grad():
        # Loop over the whole test set once
        # And compute the accuracy for the test set
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Flatten images
            images = images.view(-1, 28*28)
            # Forward pass
            output = network(images)
            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            print('Test Accuracy of the network on the {} test images: {:.2f} %'.format(total, 100 * accuracy))
        pass
test(network, test_loader)
print('pp')

