from __future__ import annotations

# Reconstruct ONE slice with a trained E2E-VarNet and save the resulting image.
# Reuses the loaders from varnet_ssim_scores.py: same checkpoint loading, same
# per-slice HDF5 read, and the same random_vds masking (min_k == max_k ==
# sampling_rate). Meant for eyeballing a single reconstruction in a notebook.

from pathlib import Path
from typing import Optional

import torch

from e2evarnet.inference.varnet_ssim_scores import (
    load_varnet,
    load_slice,
    make_masked_kspace,
)


def reconstruct_slice(
    volume_path: Path,
    slice_idx: int,
    sampling_rate: float,
    output_path: Path,
    ckpt_path: Path,
    seed: int = 100,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Undersample one slice at `sampling_rate` (min_k == max_k) and save the
    VarNet reconstruction.

    Returns the reconstructed magnitude image [H, W] and also torch.saves it to
    `output_path`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    varnet = load_varnet(ckpt_path, device)
    kspace, _target, attrs = load_slice(volume_path, slice_idx)

    # Reproducible mask draw for this (volume, slice, rate) -- mirrors the seeding
    # scheme in varnet_ssim_scores.py so the mask matches that script's.
    rate_token = int(round(sampling_rate * 100))
    mask_seed = tuple(map(ord, f"{seed}:{Path(volume_path).name}:{slice_idx}:{rate_token}"))

    masked_kspace, mask, num_low_frequencies = make_masked_kspace(
        kspace=kspace,
        attrs=attrs,
        sampling_fraction=sampling_rate,
        seed=mask_seed,
        device=device,
    )

    with torch.no_grad():
        out = varnet(masked_kspace, mask, num_low_frequencies)  # [1, H, W]

    recon = out.squeeze(0).cpu()  # [H, W]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(recon, output_path)
    print(f"Saved VarNet slice recon {tuple(recon.shape)} -> {output_path}", flush=True)
    return recon


def main() -> None:
    # --- inputs (EDIT THESE) ---
    volume_path = Path("/gpfs/scratch/shaana01/knee_fastmri_cp/multicoil_test/fileXXXX.h5")
    slice_idx = 20
    sampling_rate = 0.15
    ckpt_path = Path(
        "/gpfs/scratch/shaana01/conformal-mri-reconstruction-logs/varnet_runs/run_2/e2e_varnet/checkpoints/e2e_knee_rvds/last.ckpt"
    )
    output_path = Path(
        "/gpfs/scratch/shaana01/conformal-mri-reconstruction-logs/varnet_runs/run_2/e2e_varnet/single_slice_recons/fileXXXX_slice20_rate015.pt"
    )

    reconstruct_slice(
        volume_path=volume_path,
        slice_idx=slice_idx,
        sampling_rate=sampling_rate,
        output_path=output_path,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()


# Run from the conformal-mri-reconstruction repo root (so common.* / e2evarnet.*
# resolve; put the repo root on PYTHONPATH):
#   export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
#   python e2evarnet/inference/reconstruct_slice.py
#
# Display the saved image in a notebook:
#   import torch, matplotlib.pyplot as plt
#   img = torch.load("<output_path>")          # [H, W]
#   plt.imshow(img, cmap="gray"); plt.axis("off"); plt.show()
