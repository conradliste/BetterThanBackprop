import torch
import math
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal

reconstruction_function = torch.nn.MSELoss()
def vae_loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    # x = torch.flatten(x, start_dim=1)
    recons_loss = reconstruction_function(x,recon_x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return recons_loss, KLD

class MnistVAE(nn.Module):
    def __init__(self,latent_dims=30):
        super(MnistVAE, self).__init__()

        self.encoder = nn.Sequential(
	    nn.Linear(784, 400),
            nn.LeakyReLU(0.2),
        )
        self.fc21 = nn.Linear(400, latent_dims)
        self.fc22 = nn.Linear(400, latent_dims)

        self.decoder = nn.Sequential(
	    nn.Linear(latent_dims, 400),
            nn.LeakyReLU(0.2),
	    nn.Linear(400, 784),
	    nn.Sigmoid(),
	)

    def encode(self, x):
        x = torch.flatten(x,start_dim=1)
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def bottleneck(self, h):
        mu, logvar = self.fc21(h), self.fc22(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.decoder(z)
        return z.reshape(-1,1,28,28)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def save(self, fn_enc, fn_dec):
        torch.save(self.encoder.state_dict(), fn_enc)
        torch.save(self.decoder.state_dict(), fn_dec)

    def load(self, fn_enc, fn_dec):
        self.encoder.load_state_dict(torch.load(fn_enc))
        self.decoder.load_state_dict(torch.load(fn_dec))

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)

class MnistVAE2(nn.Module):
    def __init__(self,latent_dims=50):
        super(MnistVAE2, self).__init__()

        self.encoder = nn.Sequential(
	        nn.Conv2d(1, 64, 3, 1),
            nn.GroupNorm(min(32, 64), 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.GroupNorm(min(32, 64), 64),  
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128,1),
        )
        self.fc21 = nn.Linear(2304, latent_dims)
        self.fc22 = nn.Linear(2304, latent_dims)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,2,stride=1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode = 'nearest'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,1,3,padding=1),
            nn.Sigmoid()
	    )

    def encode(self, x):
        h = self.encoder(x)
        # z, mu, logvar = self.bottleneck(torch.flatten(h,start_dim=1))
        return h

    def bottleneck(self, h):
        mu, logvar = self.fc21(h), self.fc22(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def save(self, fn_enc, fn_dec):
        torch.save(self.encoder.state_dict(), fn_enc)
        torch.save(self.decoder.state_dict(), fn_dec)

    def load(self, fn_enc, fn_dec):
        self.encoder.load_state_dict(torch.load(fn_enc))
        self.decoder.load_state_dict(torch.load(fn_dec))

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)