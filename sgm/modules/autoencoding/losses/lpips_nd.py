import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *
from sgm.modules.autoencoding.lpips.model.model import NLayerDiscriminator3D, NLayerDiscriminator4D


class LPIPSWithDiscriminator(nn.Module):
    """
    Source: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py#L7
    """

    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        datadim=2,
        version3d=False,
        *args,
        **kwargs,
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        if perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()
        else:
            self.perceptual_loss = None
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        if datadim == 2 or (not version3d and datadim != 4):
            print("################ 2D discriminator  ###################")
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm,
            ).apply(weights_init)
        elif datadim == 3 or version3d:
            print("################ 3D discriminator  ###################")
            self.discriminator = NLayerDiscriminator3D(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm,
            ).apply(weights_init)
        elif datadim == 4:
            print("################ 4D discriminator  ###################")
            print("Currently turned off option")
            # self.discriminator = NLayerDiscriminator4D(input_nc=disc_in_channels,
            #                                            n_layers=disc_num_layers,
            #                                            use_actnorm=use_actnorm
            #                                            ).apply(weights_init)
        else:
            AttributeError(
                f"Given dimension number for data is not valid! ({datadim}) Only 2, 3 can be used!"
            )
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        global_step,
        optimizer_idx = 0,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        if posteriors is not None:
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        else:
            kl_loss = 0.0

        # now the GAN part
        if optimizer_idx == 0:
            if self.disc_factor > 0.0:
                # generator update
                if cond is None:
                    assert not self.disc_conditional
                    logits_fake = self.discriminator(reconstructions.contiguous())
                else:
                    assert self.disc_conditional
                    logits_fake = self.discriminator(
                        torch.cat((reconstructions.contiguous(), cond), dim=1)
                    )
                g_loss = -torch.mean(logits_fake)

                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            # loss = self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + rec_loss
            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
            )

            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): (
                    kl_loss.detach().mean()
                    if isinstance(kl_loss, torch.Tensor)
                    else kl_loss
                ),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/loss/rec".format(split): rec_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1 and self.disc_factor > 0.0:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log
