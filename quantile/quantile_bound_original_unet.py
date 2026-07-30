import torch
import torch.nn as nn
import torch.nn.functional as F

# from fastMRI.quantile_regression_batches.original_unet import Unet
from quantile.original_unet import Unet


class QuantileBoundOriginalUnet(nn.Module):
    """
    Vanilla U-Net with a sigmoid output head for quantile bound prediction.

    The U-Net processes the input directly (no external norm/unnorm), relying
    on its internal InstanceNorm layers for scale handling. The sigmoid output
    head ensures:
        - Upper bound: output in [input_x, 2 * input_x]
        - Lower bound: output in [0, input_x]
    """

    def __init__(
        self,
        chans: int = 32,
        num_pools: int = 4,
        in_chans: int = 1,
        out_chans: int = 1,
        drop_prob: float = 0.0,
        upper: bool = True,
        output_version: str = "v1",
    ):
        super().__init__()

        if output_version not in {"v1", "v2", "v3"}:
            raise ValueError(
                f"Invalid output_version: {output_version}. Must be 'v1', 'v2', or 'v3'."
            )

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )
        self.upper = upper
        self.output_version = output_version

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.unet(x)

        if self.output_version == "v1":
            if self.upper:
                output = torch.sigmoid(x) * input_x + input_x
            else:
                output = input_x - torch.sigmoid(x) * input_x

        elif self.output_version == "v2":
            if self.upper:
                output = x + input_x
            else:
                output = input_x - x
                
        elif self.output_version == "v3":    
            if self.upper:
                output = F.softplus(x) + input_x
            else:
                output = input_x - F.softplus(x)

        return output
