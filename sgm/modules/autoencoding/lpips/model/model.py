import functools

import torch.nn as nn

from ..util import ActNorm
from sgm.modules.conv4d import Conv4d


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaptiveInstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, latent, embed):
        latent_mean = latent.mean(dim=(2, 3, 4), keepdim=True)
        latent_std = latent.std(dim=(2, 3, 4), keepdim=True)
        embed_mean = embed.mean(dim=(2, 3, 4), keepdim=True)
        embed_std = embed.std(dim=(2, 3, 4), keepdim=True)

        normalized_latent = (latent - latent_mean) / (latent_std + self.eps)
        return normalized_latent * embed_std + embed_mean


class NLayerDiscriminator3D(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            # norm_layer = ActNorm3D
            raise NotImplementedError
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(
                input_nc,
                ndf,
                kernel_size=(kw, kw, kw),
                stride=(2, 2, 2),
                padding=(padw, padw, padw),
            ),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(1, kw, kw),
                    stride=(1, 2, 2),
                    padding=(0, padw, padw),
                    bias=True,
                ),
                nn.Conv3d(
                    ndf * nf_mult,
                    ndf * nf_mult,
                    kernel_size=(kw, 1, 1),
                    stride=(2, 1, 1),
                    padding=(padw, 0, 0),
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=(1, kw, kw),
                stride=1,
                padding=(0, padw, padw),
                bias=True,
            ),
            nn.Conv3d(
                ndf * nf_mult,
                ndf * nf_mult,
                kernel_size=(kw, 1, 1),
                stride=1,
                padding=(padw, 0, 0),
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(
                ndf * nf_mult,
                1,
                kernel_size=(kw, kw, kw),
                stride=1,
                padding=(padw, padw, padw),
            )
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class NLayerDiscriminator4D(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator4D, self).__init__()
        if not use_actnorm:
            norm_layer = functools.partial(nn.GroupNorm, num_groups=8)
        else:
            # norm_layer = ActNorm3D
            raise NotImplementedError
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.GroupNorm
        else:
            use_bias = norm_layer != nn.GroupNorm

        kw = 4
        padw = 1
        sequence = [
            Conv4d(
                input_nc,
                ndf,
                kernel_size=(3, kw, kw, kw),
                stride=(1, 2, 2, 2),
                padding=(1, padw, padw, padw),
            ),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                Conv4d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(1, 1, kw, kw),
                    stride=(1, 1, 2, 2),
                    padding=(0, 0, padw, padw),
                    bias=True,
                ),
                Conv4d(
                    ndf * nf_mult,
                    ndf * nf_mult,
                    kernel_size=(1, kw, 1, 1),
                    stride=(1, 2, 1, 1),
                    padding=(0, padw, 0, 0),
                    bias=True,
                ),
                Conv4d(
                    ndf * nf_mult,
                    ndf * nf_mult,
                    kernel_size=(kw, 1, 1, 1),
                    stride=(2, 1, 1, 1),
                    padding=(padw, 0, 0, 0),
                    bias=use_bias,
                ),
                norm_layer(num_channels=ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            Conv4d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=(1, 1, kw, kw),
                stride=(1, 1, 2, 2),
                padding=(0, 0, padw, padw),
                bias=True,
            ),
            Conv4d(
                ndf * nf_mult,
                ndf * nf_mult,
                kernel_size=(1, kw, 1, 1),
                stride=(1, 2, 1, 1),
                padding=(0, padw, 0, 0),
                bias=True,
            ),
            Conv4d(
                ndf * nf_mult,
                ndf * nf_mult,
                kernel_size=(kw, 1, 1, 1),
                stride=(2, 1, 1, 1),
                padding=(padw, 0, 0, 0),
                bias=use_bias,
            ),
            norm_layer(num_channels=ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            Conv4d(
                ndf * nf_mult,
                1,
                kernel_size=(3, 3, 3, 3),
                stride=(1, 1, 1, 1),
                padding=(1, 1, 1, 1),
            )
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


