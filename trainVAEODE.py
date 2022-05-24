import time
from types import MethodDescriptorType
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.transforms as T
import logging
from PIL import Image

import torch.utils.tensorboard as tb

import random
import math
import os
import numpy as np
from tqdm import tqdm
from VAE import *



from torchvision.utils import save_image
import torchvision
import random

from PIL import ImageDraw
from os import path
from torchdiffeq import odeint_adjoint as odeint

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc,tol):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol , atol=self.tol )
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def learning_rate_with_decay(lr,batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def val_loss(loader,ODENet,VAE,device):
    with torch.no_grad():
        ODENet.eval()
        VAE.eval()
        loss = 0
        for x,y in loader:
            output,mu,logvar = forward(x,VAE,ODENet,device)
            reconstr_loss, KL = vae_loss_function(output,x,mu,logvar)
            loss += reconstr_loss + 4 * .0025 * KL
        ODENet.train()
        VAE.train()
        return loss/len(loader)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader
def normal_kl(mu1, lv1, mu2, lv2):
    """ Computes KL loss for VAE """

    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def forward(x,VAE,ODENet,device,latent_dim=64):
    enc = VAE.encode(x)
    qz0_mean, qz0_logvar = enc[:, :latent_dim,:,:], enc[:, latent_dim:,:,:]
    # noise
    epsilon = torch.randn(qz0_mean.size()).to(device)
    # sampling codings
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
    zt = ODENet(z0).to(device)
    output = VAE.decode(zt)

    # pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
    # analytic_kl = normal_kl(qz0_mean, qz0_logvar,
    #                         pz0_mean, pz0_logvar).sum(-1)
    # kl_loss = torch.mean(analytic_kl, dim=0)

    return output,qz0_mean, qz0_logvar

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

    parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
    parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
    parser.add_argument('--nepochs', type=int, default=160)
    parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save', type=str, default='./experiment1')
    config = parser.parse_args()

    makedirs(config.save)
    logger = get_logger(logpath=os.path.join(config.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(config)

    is_odenet = config.network == 'odenet'

    if config.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif config.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    feature_layers = [ODEBlock(ODEfunc(64),tol=config.tol)] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
    VAE = MnistVAE2().to(device)
    ODENet = nn.Sequential(*feature_layers).to(device)

    # criterion = nn.CrossEntropyLoss().to(device)
    
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        config.data_aug, config.batch_size, config.batch_size
    )
    

    if config.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(config.log_dir, 'train'), flush_secs=1)
    

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        config.lr,config.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )
    reconstruction_criterion= torch.nn.MSELoss()
    optimizer = torch.optim.SGD(list(VAE.parameters()) + list(ODENet.parameters()), lr=config.lr, momentum=0.9)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()

    # training loss metric using average
    loss_meter_t = RunningAverageMeter()
    # training loss metric without KL
    meter_train = RunningAverageMeter()
    # validation loss metric without KL
    meter_valid = RunningAverageMeter()
    # list to track  training losses
    lossTrain = []
    lossTrainWithoutKL=[]
    # list to track validation losses
    lossVal = []

    factor = 0.99
    min_lr = 1e-7
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=factor, patience=5, verbose=False, threshold=1e-5,
                                                     threshold_mode='rel', cooldown=0, min_lr=min_lr, eps=1e-08)
    end = time.time()
    
    for itr in range(config.nepochs * batches_per_epoch):
        if itr == 0:
            # logger.info(model)
            logger.info('Number of parameters: {}'.format(count_parameters(ODENet)+count_parameters(VAE)))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        scheduler.step(metrics=loss_meter_t.avg)
        
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
       
        output,mu,logvar = forward(x,VAE,ODENet,device)

        reconstr_loss,KL = vae_loss_function(output,x,mu,logvar)

        loss = reconstr_loss + 4 * .00025 * KL

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss_meter_t.update(loss.item())
        meter_train.update(loss.item() - KL.item())
        lossTrain.append(loss_meter_t.avg)
        lossTrainWithoutKL.append(meter_train.avg)

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        # test_loss = val_loss(test_loader,ODENet,VAE,device)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                # train_loss = loss(train_eval_loader,device)
                # val_loss = loss(test_loader,device)
                # if val_acc > best_acc:
                torch.save({'VAE_dict': VAE.state_dict(), 'args': config}, os.path.join(config.save, 'VAE.pth'))
                torch.save({'ODE_dict': ODENet.state_dict(), 'args': config}, os.path.join(config.save, 'ODE.pth'))
                    # best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Loss {:.4f} | Train Reconstruction Loss {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, lossTrain[-1], lossTrainWithoutKL[-1]
                    )
                )

    train_logger.close()

if __name__ == '__main__':
    main()
    #python trainVAEODE.py --network odenet --adjoint True