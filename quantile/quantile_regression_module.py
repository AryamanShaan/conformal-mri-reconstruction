# TODO(recon-migration): the frozen-VarNet loader below still uses the fastMRI
# convention `VarNetModule.load_from_checkpoint(...).varnet`. For the /VarNet
# backbone (FIVarNetModule) this does NOT work -- it must be rewritten to build a
# bare E2EVarNet from ckpt["hyper_parameters"], strip the "fi_varnet." prefix, and
# load_state_dict(strict=False). NOT done yet. See docs/VARNET_FASTMRI_FILE_MAP.md.
#
# UPDATE(recon-migration): RESOLVED. VarNetModule was reworked to mirror fastMRI -- it builds
# E2EVarNet internally and calls save_hyperparameters(), so its checkpoints carry
# `hyper_parameters` and a `varnet.` prefix. The loader below now works AS-IS (no bare-load /
# prefix-strip rewrite needed). See the working import further down.
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Sequence, Union

import pytorch_lightning as pl
import torch
# from fastMRI.fastmri.data import transforms
from common import functions as transforms
# from fastMRI.fastmri.data.subsample import create_mask_for_mask_type
from common.undersampling_patterns import create_mask_for_mask_type
# TODO(recon-migration): VarNetModule import + its load_from_checkpoint loader are
# still PENDING (deferred -- see the loader TODO at the top of this file). Left
# commented until the bare-E2EVarNet load is written; the class no longer exists here.
# from fastMRI.fastmri.pl_modules import VarNetModule
# NOTE(recon-migration): use THIS repo's VarNetModule (mirrors fastMRI: internal E2EVarNet build
# + save_hyperparameters, `varnet.` prefix). load_from_checkpoint rebuilds the model from
# hyper_parameters, so `VarNetModule.load_from_checkpoint(...).varnet` below works directly.
from e2evarnet.VarNet_module import VarNetModule

# NOTE(recon-migration): QuantileBoundNormUnet dropped -- vanilla U-Net only.
# from fastMRI.quantile_regression_batches.quantile_bound_norm_unet import QuantileBoundNormUnet
# from fastMRI.quantile_regression_batches.quantile_bound_original_unet import QuantileBoundOriginalUnet
from quantile.quantile_bound_original_unet import QuantileBoundOriginalUnet


class FrozenVarNetQuantileModule(pl.LightningModule):
    """
    Train a quantile U-Net on top of a frozen pretrained VarNet.

    The data pipeline provides masked k-space, mask, num_low_frequencies, and
    target in the same format used by VarNet training. This module:

    1. Runs the frozen VarNet forward pass.
    2. Center-crops the VarNet reconstruction to the target size. (should be 320,320 need to check)
    3. Feeds the reconstruction into the quantile U-Net.
    4. Optimizes pinball loss against the target image.
    """

    def __init__(
        self,
        varnet_checkpoint: Union[str, Path],
        quantile: float,
        pinball_loss_type: str = "original", # or normalize_by_gt_y (which does not add value, as L1 loss magnitude does not affect gradient)
        model_type: str = "QuantileBoundNormUnet", # or QuantileBoundOriginalUnet 
        output_version: str = "v1", # output head variant for the quantile U-Net, see model code
        unet_chans: int = 32,
        unet_pools: int = 4,
        unet_drop_prob: float = 0.0,
        mask_type: str = "random_vds",
        min_k_fraction: float = 0.05, # used for random_vds
        max_k_fraction: float = 0.25, # used for random_vds
        center_fractions: Optional[Sequence[float]] = None, # not used in random_vds but required for the function/ class it inherits from 
        accelerations: Optional[Sequence[int]] = None, # not used in random_vds but required for the function/ class it inherits from 
        lr: float = 3e-4, # value from knee leaderboard for training varnet. just kept the same.
        lr_step_size: int = 40, # value from knee leaderboard for training varnet. just kept the same.
        lr_gamma: float = 0.1, # value from knee leaderboard for training varnet. just kept the same.
        weight_decay: float = 0.0, # value from knee leaderboard for training varnet. just kept the same.
    ):
        super().__init__()
        self.save_hyperparameters()

        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be strictly between 0 and 1.")
        if quantile == 0.5:
            raise ValueError("quantile=0.5 is not supported for upper/lower bound training.")
        if pinball_loss_type not in {"original", "normalize_by_gt_y"}:
            raise ValueError(
                "pinball_loss_type must be either 'original' or 'normalize_by_gt_y'."
            )
        if model_type not in {"QuantileBoundNormUnet", "QuantileBoundOriginalUnet"}:
            raise ValueError(
                f"model_type must be 'QuantileBoundNormUnet' or 'QuantileBoundOriginalUnet', got '{model_type}'."
            )

        self.quantile = float(quantile)
        self.pinball_loss_type = pinball_loss_type
        self.model_type = model_type
        self.output_version = output_version
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.upper = self.quantile > 0.5  # True if training an upper 1 quantile, False if training a lower quantile

        varnet_module = VarNetModule.load_from_checkpoint(
            str(varnet_checkpoint), map_location="cpu"
        )
        self.varnet = varnet_module.varnet
        self.varnet.eval()
        for param in self.varnet.parameters():
            param.requires_grad = False

        # NOTE(recon-migration): QuantileBoundNormUnet dropped -- vanilla U-Net only.
        # Original if/else kept below (commented); now always QuantileBoundOriginalUnet.
        # if model_type == "QuantileBoundNormUnet":
        #     self.quantile_unet = QuantileBoundNormUnet(
        #         chans=unet_chans, num_pools=unet_pools, in_chans=1, out_chans=1,
        #         drop_prob=unet_drop_prob, upper=self.upper, output_version=output_version,
        #     )
        # else:
        self.quantile_unet = QuantileBoundOriginalUnet(
            chans=unet_chans,
            num_pools=unet_pools,
            in_chans=1,
            out_chans=1,
            drop_prob=unet_drop_prob,
            upper=self.upper,
            output_version=output_version,
        )

        if center_fractions is None:
            center_fractions = [0.08] # not used for random_vds but required for the function/ class it inherits from
        if accelerations is None:
            accelerations = [4] # not used for random_vds but required for the function/ class it inherits from

        # This mask function is used inside forward() so one random-VDS mask is
        # applied to the full local batch. Because apply_mask collapses both the
        # batch and coil dimensions when it builds the mask shape, the same mask
        # is broadcast across:
        #   - all slices in the local batch
        #   - all coils of each slice
        self.mask_func = create_mask_for_mask_type(
            mask_type,
            center_fractions,
            accelerations,
            min_k_fraction=min_k_fraction,
            max_k_fraction=max_k_fraction,
        )

    def pinball_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - prediction
        base_loss = torch.maximum(self.quantile * error, (self.quantile - 1.0) * error)

        if self.pinball_loss_type == "original":
            loss = base_loss.mean()
        # elif self.pinball_loss_type == "normalize_by_gt_y":
        #     # Scale the per-pixel pinball loss by the ground-truth pixel value
        #     # to optimize a relative-error variant of the quantile objective.
        #     epsilon = 1e-9
        #     loss = (base_loss / (target + epsilon)).mean()
        else:
            raise ValueError(
                f"Invalid pinball_loss_type: {self.pinball_loss_type}. Must be 'original' or 'normalize_by_gt_y'."
            )

        return loss

    def _make_validation_seed(              # IS NOT USED as long as use_seed=False in validation and train forward
        self, fnames, slice_nums: torch.Tensor
    ) -> tuple[int, ...]:
        """
        Build a deterministic batch-level seed for validation (or training, which should not be done).

        Validation should be repeatable across epochs, so we derive a seed from
        the batch contents. Training uses seed=None for fresh randomness.
        """
        tokens = []
        for fname, slice_num in zip(fnames, slice_nums.tolist()):
            tokens.append(f"{fname}:{slice_num}")
        seed_string = "|".join(tokens)
        return tuple(map(ord, seed_string))

    def forward(
        self,
        kspace: torch.Tensor,
        target: torch.Tensor,
        fnames=None,
        slice_nums: Optional[torch.Tensor] = None,
        use_seed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seed = None
        if use_seed:
            if fnames is None or slice_nums is None:
                raise ValueError("Validation forward requires fnames and slice_nums for deterministic masking.")
            seed = self._make_validation_seed(fnames, slice_nums)

        mask_shape = (1,) * len(kspace.shape[:-3]) + tuple(kspace.shape[-3:]) 
        # TODO need to confirm that kspace shape is (B, C, 640, 640, 2) specifically
        # TODO check that target shape is (B, 640, 640) specifically (i.e. no coil dimension and no complex dimension)
        # TODO check that mask_shape should be = (1,) * 2 + (640, 640, 2) = (1, 1, 640, 640, 2)
        mask, num_low_frequencies = self.mask_func(mask_shape, offset=None, seed=seed)
        # TODO check if is this mask_shape is correct for mask_func. Actually, it should be correct check /fastmri/data/transforms.py:
        # apply_mask does shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:]) followed by mask, num_low_frequencies = mask_func(shape, offset, seed)
        mask = mask.to(kspace.device).to(torch.bool)
        masked_kspace = kspace * mask + 0.0 # the + 0.0 removes the sign of the zeros

        self.varnet.eval()
        with torch.no_grad():
            varnet_recon = self.varnet(masked_kspace, mask, num_low_frequencies)

        target_cropped, varnet_recon_cropped = transforms.center_crop_to_smallest(
            target, varnet_recon
        ) # TODO check that size post cropping is (B, 320, 320) for both target and varnet_recon, 
        # hence need unsqueeze(1) to add channel dimension before feeding into unet


        unet_input = varnet_recon_cropped.unsqueeze(1)
        quantile_prediction = self.quantile_unet(unet_input)

        return quantile_prediction, target_cropped.unsqueeze(1), unet_input

    def training_step(self, batch, batch_idx):
        prediction, target, _ = self.forward(
            batch.kspace,
            batch.target,
            use_seed=False,
        )
        loss = self.pinball_loss(prediction, target)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=target.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, target, _ = self.forward(
            batch.kspace,
            batch.target,
            fnames=batch.fname,
            slice_nums=batch.slice_num,
            use_seed=False,
        )
        loss = self.pinball_loss(prediction, target)
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=target.shape[0],
        )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.quantile_unet.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--varnet_checkpoint",
            type=Path,
            required=False,
            default=None,
            help="Path to the frozen pretrained VarNet checkpoint.",
        )
        parser.add_argument(
            "--quantile",
            type=float,
            default=0.95,
            help="Quantile level to train, e.g. 0.95 for upper bound or 0.05 for lower bound.",
        )
        parser.add_argument(
            "--pinball_loss_type",
            choices=("original", "normalize_by_gt_y"),
            default="original",
            type=str,
            help=(
                "Pinball loss variant: 'original' uses the standard objective, "
                "while 'normalize_by_gt_y' divides each pixel loss by the target pixel value."
            ),
        )
        parser.add_argument(
            "--model_type",
            choices=("QuantileBoundNormUnet", "QuantileBoundOriginalUnet"),
            default="QuantileBoundNormUnet",
            type=str,
            help="Which quantile U-Net architecture to use.",
        )
        parser.add_argument(
            "--output_version",
            default="v1",
            type=str,
            help="Output head variant. QuantileBoundNormUnet supports v1-v4; QuantileBoundOriginalUnet supports v1-v2.",
        )
        parser.add_argument(
            "--unet_chans",
            type=int,
            default=32,
            help="Base number of channels in the quantile U-Net.",
        )
        parser.add_argument(
            "--unet_pools",
            type=int,
            default=4,
            help="Number of pooling layers in the quantile U-Net.",
        )
        parser.add_argument(
            "--unet_drop_prob",
            type=float,
            default=0.0,
            help="Dropout probability for the quantile U-Net.",
        )
        parser.add_argument(
            "--mask_type",
            choices=("random", "equispaced", "random_vds"),
            default="random_vds",
            type=str,
            help="Type of k-space mask to apply inside the frozen VarNet pipeline.",
        )
        parser.add_argument(
            "--min_k_fraction",
            type=float,
            default=0.05,
            help="Minimum fraction of k-space to sample for random_vds.",
        )
        parser.add_argument(
            "--max_k_fraction",
            type=float,
            default=0.25,
            help="Maximum fraction of k-space to sample for random_vds.",
        )
        parser.add_argument(
            "--center_fractions",
            nargs="+",
            default=[0.08],
            type=float,
            help="Center fractions passed to the mask factory.",
        )
        parser.add_argument(
            "--accelerations",
            nargs="+",
            default=[4],
            type=int,
            help="Accelerations passed to the mask factory.",
        )
        parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
        parser.add_argument(
            "--lr_step_size",
            type=int,
            default=40,
            help="Epoch at which to decrease the learning rate.",
        )
        parser.add_argument(
            "--lr_gamma",
            type=float,
            default=0.1,
            help="Learning-rate decay factor.",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="Weight decay for the U-Net optimizer.",
        )
        return parser
