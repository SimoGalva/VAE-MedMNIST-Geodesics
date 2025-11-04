from collections import OrderedDict

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class UnflattenDecoder(nn.Module):
    def __init__(self):
        super(UnflattenDecoder, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size,64, 4, 4)

class WeightSigmoid (nn.Module):
    def __init__(self, weight: float = 1):
        super(WeightSigmoid, self).__init__()
        self.k = weight
    def forward(self, x):
        return torch.sigmoid(self.k * x)

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w-8)//2 - 4)//2
        hh = ((h-8)//2 - 4)//2
        self.encode = nn.Sequential(nn.Conv2d(c, 16, 5, padding = 0), nn.BatchNorm2d(16), nn.ReLU(inplace = True),
                                    nn.Conv2d(16, 32, 5, padding = 0), nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                    nn.MaxPool2d(2, 2),
                                    Flatten(), MLP([ww*hh*64, 256, 128])
                                   )
        self.calc_mean = MLP([128+ncond, 64, nhid], last_activation = False)
        self.calc_logvar = MLP([128+ncond, 64, nhid], last_activation = False)
    def forward(self, x, y = None):
        x = self.encode(x)
        if (y is None):
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))

class Decoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.nhid = nhid
        self.shape = shape
        self.decode = nn.Sequential(MLP([nhid, 256, 128, nhid*64], last_activation = True), UnflattenDecoder(),
                                    nn.Upsample(scale_factor=2, mode='bilinear'),
                                    nn.ConvTranspose2d(64, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                    nn.ConvTranspose2d(64, 32, 3, padding = 0), nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                                    nn.Upsample(scale_factor=2, mode='bilinear'),
                                    nn.ConvTranspose2d(32, 16, 5, padding = 0), nn.BatchNorm2d(16), nn.ReLU(inplace = True),
                                    nn.ConvTranspose2d(16, c, 5, padding = 0), nn.BatchNorm2d(c), nn.ReLU(inplace = True),
                                    Flatten(), MLP([32*32, 64, 128, 256, c*w*h], last_activation = False), WeightSigmoid(weight = 0.5)
                                    )
    def forward(self, z, y = None):
        c, w, h = self.shape
        batch_size = z.shape[0]
        z = z.view(batch_size, self.nhid)
        if (y is None):
            return self.decode(z).view(-1, c, w, h)
        else:
            return self.decode(torch.cat((z, y), dim=1)).view(-1, c, w, h)

class VAE(nn.Module):
    def __init__(self, shape, nhid = 16):
        super(VAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid)
        self.decoder = Decoder(shape, nhid)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar

    def generate(self, batch_size = None):
        z = torch.randn((batch_size, self.dim)) if batch_size else torch.randn((1, self.dim))
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res

    def getDecoder(self):
      return self.decoder

    def getEncoder(self):
      return self.encoder

lossFunc = nn.MSELoss(reduction = "sum")
def loss(X, X_hat, mean, logvar):
    reconstruction_loss = lossFunc(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence