import torch
from unet import UNet, ConvPass
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

        self.mask_head = ConvPass(num_fmaps,1,[[1,1]],activation='Sigmoid')

    def forward(self,input):

        z = self.unet(input)
        mask = self.mask_head(z)

        return mask


class WeightedBCELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedBCELoss, self).__init__()

#    def _calc_loss(self, pred, target, weights):
#
#        m = torch.nn.Sigmoid()
#        pred = m(pred)
#
#        scale = - weights * target * torch.log(pred) - (1 - target) * weights * torch.log(1 - pred)
#
#        if len(torch.nonzero(scale)) != 0:
#
#            loss = torch.mean(
#                    torch.masked_select(
#                        scale,
#                        torch.gt(weights, 0)
#                        )
#                    )
#
#        else:
#
#            loss = torch.mean(scale)
#
#        return loss

    def forward(
            self,
            pred,
            target,
            weights):

        loss = torch.nn.functional.binary_cross_entropy(pred,target,weights)

        return loss
