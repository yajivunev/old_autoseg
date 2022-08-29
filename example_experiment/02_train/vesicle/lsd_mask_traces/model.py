import torch
from funlib.learn.torch.models import UNet, ConvPass
from gunpowder import Coordinate, Roi
import numpy as np

class UnetModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up):

        super().__init__()

        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up)

        self.lsd_head = ConvPass(num_fmaps,6,[[1,1]],activation='Sigmoid')
        self.mask_head = ConvPass(num_fmaps,1,[[1,1]],activation='Sigmoid')

    def forward(self,input):

        z = self.unet(input)
        lsd = self.lsd_head(z)
        mask = self.mask_head(z)

        return lsd,mask

class WeightedLoss(torch.nn.Module):

    def __init__(self):
        super(WeightedLoss, self).__init__()

    def _calc_mse(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            loss = torch.mean(
                    torch.masked_select(
                        scale,
                        torch.gt(weights, 0)
                        )
                    )

        else:

            loss = torch.mean(scale)

        return loss

    def _calc_bce(self, pred, target, weights):

        scale = - weights * target * torch.log(pred) - (1 - target) * weights * torch.log(1 - pred)

        if len(torch.nonzero(scale)) != 0:

            loss = torch.mean(
                    torch.masked_select(
                        scale,
                        torch.gt(weights, 0)
                        )
                    )

        else:

            loss = torch.mean(scale)

        return loss

    def forward(
            self,
            pred_lsd,
            target_lsd,
            lsd_weights,
            pred_mask,
            target_mask,
            mask_weights):

        lsd_loss = self._calc_mse(pred_lsd, target_lsd, lsd_weights)
        mask_loss = self._calc_bce(pred_mask, target_mask, mask_weights)

        return lsd_loss + mask_loss
