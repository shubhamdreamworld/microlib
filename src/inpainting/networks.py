import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


def make_nets_rect(config, training=True):
    """Creates Generator and Discriminator class objects from params either loaded from config object or params file.

    :param config: a Config class object 
    :type config: Config
    :param training: if training is True, params are loaded from Config object. If False, params are loaded from file, defaults to True
    :type training: bool, optional
    :return: Discriminator and Generator class objects
    :rtype: Discriminator, Generator
    """
    # save/load params
    if training:
        config.save()
    else:
        config.load()

    dk, ds, df, dp, gk, gs, gf, gp = config.get_net_params()

    # Make nets
    if config.net_type == 'gan':
        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()
                for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
                    self.convs.append(nn.ConvTranspose2d(
                        gf[lay], gf[lay+1], k, s, p, bias=False))
                    self.bns.append(nn.BatchNorm2d(gf[lay+1]))

            def forward(self, x):
                # x = torch.cat((x, mask[:,-1].reshape(x.shape)), dim=1)
                for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
                    x = F.relu_(bn(conv(x)))
                if config.image_type == 'n-phase':
                    out = torch.softmax(self.convs[-1](x), dim=1)
                else:
                    out = torch.sigmoid(self.convs[-1](x))  # bs x 1 x 1
                # out = torch.where((mask[:,-1]==0).unsqueeze(1).repeat(1,3,1,1,1), mask[:,0:3], out)
                return out  # bs x n x imsize x imsize x imsize

        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()
                self.convs = nn.ModuleList()
                for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                    self.convs.append(
                        nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

            def forward(self, x):
                for conv in self.convs[:-1]:
                    x = F.relu_(conv(x))
                return x
    else:
        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()
                self.up = nn.Upsample(scale_factor=2)

                for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
                    self.convs.append(nn.Conv2d(
                        gf[lay], gf[lay+1], 3, 1, 1, bias=False, padding_mode='reflect'))
                    self.bns.append(nn.BatchNorm2d(gf[lay+1]))

            def forward(self, x):
                # x = torch.cat((x, mask[:,-1].reshape(x.shape)), dim=1)
                for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
                    x = F.relu_(bn(self.up(conv(x))))
                if config.image_type == 'n-phase':
                    out = torch.softmax(self.convs[-1](x), dim=1)
                else:
                    out = torch.sigmoid(self.convs[-1](x))
                # out = torch.where((mask[:,-1]==0).unsqueeze(1).repeat(1,3,1,1,1), mask[:,0:3], out)
                return out  # bs x n x imsize x imsize x imsize

        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()
                self.convs = nn.ModuleList()
                for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                    self.convs.append(
                        nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

            def forward(self, x):
                for conv in self.convs[:-1]:
                    x = F.relu_(conv(x))
                x = self.convs[-1](x)  # bs x 1 x 1
                return x

    return Discriminator, Generator

def make_nets_poly(config, training=True):
    """Creates Generator and Discriminator class objects from params either loaded from config object or params file.

    :param config: a Config class object 
    :type config: Config
    :param training: if training is True, params are loaded from Config object. If False, params are loaded from file, defaults to True
    :type training: bool, optional
    :return: Discriminator and Generator class objects
    :rtype: Discriminator, Generator
    """
    # save/load params
    if training:
        config.save()
    else:
        config.load()

    dk, ds, df, dp, gk, gs, gf, gp = config.get_net_params()
    df[0] = config.n_phases
    gf[-1] = config.n_phases
    # Make nets
    if config.net_type == 'gan':
        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()
                for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
                    self.convs.append(nn.ConvTranspose2d(
                        gf[lay], gf[lay+1], k, s, p, bias=False))
                    self.bns.append(nn.BatchNorm2d(gf[lay+1]))

            def forward(self, x):
                # x = torch.cat((x, mask[:,-1].reshape(x.shape)), dim=1)
                for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
                    x = F.relu_(bn(conv(x)))
                if config.image_type == 'n-phase':
                    out = torch.softmax(self.convs[-1](x), dim=1)
                else:
                    out = torch.sigmoid(self.convs[-1](x))
                # out = torch.where((mask[:,-1]==0).unsqueeze(1).repeat(1,3,1,1,1), mask[:,0:3], out)
                return out  # bs x n x imsize x imsize x imsize
    else:
        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.convs = nn.ModuleList()
                self.bns = nn.ModuleList()
                self.up = nn.Upsample(scale_factor=2)

                for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
                    self.convs.append(nn.Conv2d(
                        gf[lay], gf[lay+1], 3, 1, 1, bias=False, padding_mode='reflect'))
                    self.bns.append(nn.BatchNorm2d(gf[lay+1]))

            def forward(self, x):
                # x = torch.cat((x, mask[:,-1].reshape(x.shape)), dim=1)
                for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
                    x = F.relu_(bn(self.up(conv(x))))
                if config.image_type == 'n-phase':
                    out = torch.softmax(self.convs[-1](x), dim=1)
                else:
                    out = torch.sigmoid(self.convs[-1](x))
                # out = torch.where((mask[:,-1]==0).unsqueeze(1).repeat(1,3,1,1,1), mask[:,0:3], out)
                return out  # bs x n x imsize x imsize x imsize

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(
                    nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)  # bs x 1 x 1
            return x

    return Discriminator, Generator
