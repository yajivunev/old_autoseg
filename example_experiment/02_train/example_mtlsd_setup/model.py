import torch
from funlib.learn.torch.models import UNet, ConvPass

class MtlsdModel(torch.nn.Module):

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

        self.lsd_head = ConvPass(num_fmaps,10,[[1,1,1]],activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps,3,[[1,1,1]],activation='Sigmoid')

    def forward(self,input):

        z = self.unet(input)
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        return lsds,affs

class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, lsds_prediction, lsds_target, lsds_weights, affs_prediction, affs_target, affs_weights,):

        loss1 = super(WeightedMSELoss, self).forward(
                lsds_prediction*lsds_weights,
                lsds_target*lsds_weights)

        loss2 = super(WeightedMSELoss, self).forward(
            affs_prediction*affs_weights,
            affs_target*affs_weights)

        return loss1 + loss2
