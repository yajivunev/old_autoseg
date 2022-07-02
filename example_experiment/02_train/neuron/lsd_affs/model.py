import torch
from funlib.learn.torch.models import UNet, ConvPass
from gunpowder import Coordinate, Roi
import numpy as np
from pytorch_lightning import LightningModule
from apex.optimizers import FusedAdam

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

        self.aff_head = ConvPass(num_fmaps,3,[[1,1,1]],activation='Sigmoid')
        self.lsd_head = ConvPass(num_fmaps,10,[[1,1,1]],activation='Sigmoid')

    def forward(self,input):

        z = self.unet(input)
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        return lsds,affs


class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):

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

    def forward(
            self,
            lsds_prediction,
            lsds_target,
            lsds_weights,
            affs_prediction,
            affs_target,
            affs_weights):

        loss1 = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)
        loss2 = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return loss1 + loss2


#util functions
def calc_max_padding(
        output_size,
        voxel_size,
        neighborhood=None,
        sigma=None,
        mode='shrink'):

    '''Calculate maximum labels padding needed.
    Args:
        output_size (array-like of ``int``):
            output size of network, in world units (a gunpowder coordinate)
        voxel_size (array-like of ``int``):
            voxel size to use (a gunpowder coordinate)
        neighborhood (``list`` of array-like, optional):
            affinity neighborhood to use.
        sigma (``int``, optional):
            sigma if using lsds
        mode (``string``, optional):
            mode to use for snapping roi to grid, see gunpowder roi
            documentation for details
    Explanation:
        when padding labels, we need to ensure that each batch still contains at
        least 50% of GT data. Additionally, we need to also consider worst case
        45 degree rotation when elastically augmenting the data. Our max padding
        is calculated as follows:
            output_size = output size of network in world coordinates (i.e \
                    nanometers not voxels)
            method_padding = largest affinity neighborhood * voxel size (for \
                    affinities) or sigma * voxel size (for lsds)
            diagonal = diagonal between x and y dimensions (i.e square root \ of
            the sum of squares of x and y axes)
            max_padding = (output_size[z]/2, diagonal/2, diagonal/2) + \
                    method_padding
        we then need to ensure max padding is a multiple of the voxel size - use
        snap_to_grid for this (see gunpowder.roi.snap_to_grid())
    '''

    if neighborhood is not None:

        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = Coordinate(
                            [np.abs(aff) for val in neighborhood \
                                    for aff in val if aff != 0]
                        )

        method_padding = voxel_size * max_affinity

    if sigma:

        method_padding = Coordinate((sigma*3,)*3)

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = Roi(
                    (Coordinate(
                        [i/2 for i in [output_size[0], diag, diag]]) +
                        method_padding),
                    (0,)*3).snap_to_grid(voxel_size,mode=mode)

    return max_padding.get_begin()

class MtLsdModule(LightningModule):

    def __init__(self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            batch_size,
            input_shape):

        super().__init__()
        self.unet = MtlsdModel(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up)

        self.example_input_array = torch.rand([batch_size]+[1]+input_shape)
        self.loss_fn = WeightedMSELoss()

    def configure_optimizers(self):

        optimizer = FusedAdam(
            self.unet.parameters(),
            lr=0.5e-4,
            betas=(0.95,0.999),
            eps=1e-08)

        return optimizer

    def forward(self,input):

        lsds,affs = self.unet(input)

        return lsds,affs

    def training_epoch_end(self,outputs):

        #add histograms
        for name,params in self.named_parameters():
        
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

        #compute epoch-level loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss_lsds = torch.stack([x['lsds_loss'] for x in outputs]).mean()
        avg_loss_affs = torch.stack([x['affs_loss'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("epoch_loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("epoch_loss_lsds", avg_loss_lsds, self.current_epoch)
        self.logger.experiment.add_scalar("epoch_loss_affs", avg_loss_affs, self.current_epoch)

    def training_step(self,batch,batch_idx):

        opt = self.optimizers()
        opt.zero_grad()

        raw,gt_lsds,lsds_weights,gt_affs,affs_weights,batch_id = batch

        pred_lsds,pred_affs = self.unet(raw)
        lsds_loss = self.loss_fn._calc_loss(pred_lsds, gt_lsds, lsds_weights)
        affs_loss = self.loss_fn._calc_loss(pred_affs, gt_affs, affs_weights)

        loss = lsds_loss + affs_loss

        self.logger.experiment.add_scalar("loss",loss,self.global_step) 
        self.logger.experiment.add_scalar("lsds_loss",loss,self.global_step) 
        self.logger.experiment.add_scalar("affs_loss",loss,self.global_step) 

        batch_dictionary = {
            'loss': loss,
            'lsds_loss': lsds_loss,
            'affs_loss': affs_loss,
            'batch_idx': batch_idx,
            'batch.id': batch_id,
            #TO DO: add VOI/NVI",
            'global_step': self.global_step}

        return batch_dictionary
