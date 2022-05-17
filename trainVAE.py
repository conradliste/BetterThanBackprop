import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.transforms as T
from PIL import Image

import torch.utils.tensorboard as tb

import random
import math
import os
from tqdm import tqdm
from VAE import *



from torchvision.utils import save_image
import torchvision
import random

from PIL import ImageDraw
from os import path

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

def main():
    torch.cuda.empty_cache()
    import argparse

    parser = argparse.ArgumentParser()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data', type=str, default='MNIST')
    
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--log_dir', type=str, default='.')
    


    config = parser.parse_args()

    if config.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(config.log_dir, 'train'), flush_secs=1)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])), 
        batch_size=config.batch_size, shuffle=True)
    
    VAE = MnistVAE(latent_dims=2).to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr=1e-3)
    transform = T.ToPILImage()

    for epoch in range(config.epochs+1):
        if config.data == "MNIST":
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('./data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                            ])),
                batch_size=config.batch_size, shuffle=True,num_workers=1)
            epoch_length = len(train_loader)
            
            for batch_index, (img_batch,label) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',total=epoch_length):
                img_batch = img_batch.to(device)
                optimizer.zero_grad()
                z,mu,logvar = VAE(img_batch)
                loss = vae_loss_function(z,img_batch,mu,logvar)
                
                if train_logger is not None:
                    if epoch_length % config.batch_size == batch_index - 1:
                        # z = torch.reshape(z, (config.batch_size,1,28,28))
                        train_logger.add_images('reconstructed images',z,  batch_index + epoch*epoch_length )
                        train_logger.add_scalar('VAE Loss', loss, batch_index + epoch*epoch_length)

                loss.backward()
                optimizer.step()

            # if epoch % 50 == 0:
                # VAE.save(f"enc_weights/enc_{epoch}.pt",f"dec_weights/dec_{epoch}.pt")
    train_logger.close()

if __name__ == '__main__':
    main()