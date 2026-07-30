from __future__ import annotations

# Build per-slice quantile bounds over a NESTED sweep of sampling rates.
#
# For every volume in a calibration set, for every slice, we run an incremental
# (monotonically nested) variable-density undersampling sweep:
#
#     rate_0 (e.g. 0.05):  mask_0 = IncrementalVDS(target_k_fraction=rate_0)
#     rate_1 (e.g. 0.10):  mask_1 = IncrementalVDS(target_k_fraction=rate_1,
#                                                  previous_mask=mask_0)   # mask_1 >= mask_0
#     ...
#
# At each rate we:
#   1. mask the raw multicoil k-space with the (nested) mask,
#   2. reconstruct with the frozen VarNet,
#   3. feed the (cropped) VarNet reconstruction into BOTH the upper- and
#      lower-quantile U-Nets to get pixelwise prediction-interval bounds.
#
# This reproduces the exact forward path of FrozenVarNetQuantileModule.forward
# (quantile_regression_module.py): the mask is applied directly as kspace * mask
# (NO acquisition-padding clip -- training did the same), the VarNet recon is
# center-cropped to the target, then unsqueezed to [B,1,H,W] before the U-Net.
# The only deliberate change vs training is the mask FAMILY: training used a
# fresh RandomVariableDensityMaskFunc per forward; here we use the nested
# IncrementalVariableDensityMaskFunc so the higher-rate acquisitions physically
# contain the lower-rate ones.
#
# Checkpoint loading:
#   * VarNet: a leaderboard Lightning VarNetModule .ckpt -- weights under
#     state_dict["varnet.*"], arch in ckpt["hyper_parameters"]. Loaded into a
#     bare fastmri VarNet (no Lightning import), same as varnet_ssim_score_2.py.
#   * Quantile U-Nets: FrozenVarNetQuantileModule .ckpt -- the trained U-Net
#     weights live under state_dict["quantile_unet.*"], arch/quantile in
#     ckpt["hyper_parameters"]. We rebuild a bare QuantileBoundOriginalUnet
#     (model_type in the batch_14 configs) and load only that sub-state_dict,
#     matching export_slice_quantile_bounds.py's convention.
#
# Output (.pt): a nested dict keyed by integer volume id ->
#   results[vol_id] = {
#       "volume_name": <str>,
#       slice_idx: {
#           rate: {
#               "target_rss":     [H, W]  float32,  ground-truth RSS (cropped),
#               "mask":           [W]     bool,     column sampling pattern used,
#               "varnet_recon":   [H, W]  float32,  VarNet reconstruction (cropped),
#               "upper_quantile": [H, W]  float32,  upper-bound U-Net output,
#               "lower_quantile": [H, W]  float32,  lower-bound U-Net output,
#           },
#           ...
#       },
#       ...
#   }

# --- import shim ---------------------------------------------------------
# This repo's modules import themselves under the package name `fastMRI`
# (e.g. `from fastMRI.fastmri...`, `from fastMRI.quantile_regression_batches...`),
# which assumes the repo directory is literally named `fastMRI`. On the cluster
# the repo dir is named differently (e.g. `varnet_knee`), so `import fastMRI`
# fails. We alias `fastMRI` to the repo root (two levels up from this file) so
# every `fastMRI.*` import resolves no matter what the repo dir is called. This
# lets the script run with a plain `PYTHONPATH=<repo root>` and no symlink.
# NOTE(recon-migration): import shim no longer needed -- repo uses proper
# common/e2evarnet/quantile packages on PYTHONPATH. Whole shim commented out.
# import sys as _sys
# import types as _types
# from pathlib import Path as _Path
#
# _REPO_ROOT = str(_Path(__file__).resolve().parents[2])
# if _REPO_ROOT not in _sys.path:
#     _sys.path.insert(0, _REPO_ROOT)
# if "fastMRI" not in _sys.modules:
#     _fastmri_pkg = _types.ModuleType("fastMRI")
#     _fastmri_pkg.__path__ = [_REPO_ROOT]
#     _sys.modules["fastMRI"] = _fastmri_pkg
# -------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import torch

# from fastMRI.fastmri.data import transforms
from common import functions as transforms
# from fastMRI.fastmri.models import VarNet
from e2evarnet.VarNet import E2EVarNet as VarNet

# from fastMRI.quantile_regression_batches.quantile_bound_original_unet import (
#     QuantileBoundOriginalUnet,
# )
from quantile.quantile_bound_original_unet import QuantileBoundOriginalUnet
# from fastMRI.quantile_regression_batches.inference.incremental_mask import (
#     IncrementalVariableDensityMaskFunc,
# )
from quantile.inference.incremental_mask import IncrementalVariableDensityMaskFunc


def load_varnet(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a leaderboard-trained fastMRI VarNet from a Lightning .ckpt.

    The Lightning VarNetModule stores the network under 'varnet.' in state_dict
    and its constructor args in ckpt['hyper_parameters']. We rebuild the bare
    fastmri VarNet from those and load the stripped state_dict (no PL import).
    Identical to varnet_ssim_score_2.py.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    hp = ckpt.get("hyper_parameters", {}) or {}

    model = VarNet(
        num_cascades=hp.get("num_cascades", 12),
        sens_chans=hp.get("sens_chans", 8),
        sens_pools=hp.get("sens_pools", 4),
        chans=hp.get("chans", 18),
        pools=hp.get("pools", 4),
    )

    vsd = {k[len("varnet."):]: v for k, v in state.items() if k.startswith("varnet.")}
    if not vsd:  # already a bare varnet state_dict
        vsd = state
    missing, unexpected = model.load_state_dict(vsd, strict=False)
    print(f"load_varnet: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model.eval()
    model.to(device)
    return model


def load_quantile_unet(
    checkpoint_path: Path, device: torch.device
) -> Tuple[QuantileBoundOriginalUnet, float]:
    """Load one quantile U-Net from a FrozenVarNetQuantileModule .ckpt.

    Only the trained U-Net lives under state_dict['quantile_unet.*']; the frozen
    VarNet sub-state is ignored (we load our own VarNet separately). Arch and the
    quantile level (which sets upper vs lower via quantile>0.5) come from
    ckpt['hyper_parameters']. Mirrors export_slice_quantile_bounds.py, but for
    the QuantileBoundOriginalUnet used by the batch_14 configs.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    hp = ckpt.get("hyper_parameters", {}) or {}

    quantile = float(hp.get("quantile", 0.5))
    upper = quantile > 0.5

    model = QuantileBoundOriginalUnet(
        chans=hp.get("unet_chans", 32),
        num_pools=hp.get("unet_pools", 4),
        in_chans=1,
        out_chans=1,
        drop_prob=hp.get("unet_drop_prob", 0.0),
        upper=upper,
        output_version=hp.get("output_version", "v1"),
    )

    qsd = {
        k[len("quantile_unet."):]: v
        for k, v in state.items()
        if k.startswith("quantile_unet.")
    }
    if not qsd:  # already a bare quantile-unet state_dict
        qsd = state
    missing, unexpected = model.load_state_dict(qsd, strict=False)
    print(
        f"load_quantile_unet(quantile={quantile}, upper={upper}): "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

    model.eval()
    model.to(device)
    return model, quantile


def load_slice(h5_path: Path, slice_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read one raw multicoil slice + its RSS target (no padding needed here:
    the incremental mask is applied directly, exactly as the quantile module's
    forward does)."""
    with h5py.File(h5_path, "r") as hf:
        kspace = hf["kspace"][slice_index]
        target = hf["reconstruction_rss"][slice_index]
    return transforms.to_tensor(kspace), transforms.to_tensor(target)


def main() -> None:
    # --- inputs (EDIT THESE) ---
    # Calibration set: one or more directories of multicoil .h5 files. All .h5
    # files across every listed directory are processed.
    calib_dirs = [
        Path("/gpfs/scratch/shaana01/knee_fastmri_cp/multicoil_test"),
        # Path("/gpfs/scratch/shaana01/knee_fastmri_cp/multicoil_calibration"),
        # Path("/gpfs/scratch/shaana01/knee_fastmri_cp/multicoil_val"),
    ]

    # Frozen VarNet backbone (leaderboard Lightning .ckpt).
    varnet_checkpoint = Path(
        "/gpfs/scratch/shaana01/varnet_knee_root_dir/run_1/checkpoints/epochepoch=33-val_lossvalidation_loss=0.1117.ckpt"
    )
    # Quantile U-Nets, batch_14 (QuantileBoundOriginalUnet, output_version v1).
    # quantile=0.95 config -> upper bound; quantile=0.05 config -> lower bound.
    # Swap last.ckpt for the best 'epoch..-val_loss..ckpt' if you prefer.
    upper_checkpoint = Path(
        "/gpfs/scratch/shaana01/quantile_regression_root/quantile_0.95_batches_14/checkpoints/last.ckpt"
    )
    lower_checkpoint = Path(
        "/gpfs/scratch/shaana01/quantile_regression_root/quantile_0.05_batches_14/checkpoints/last.ckpt"
    )

    output_pt = Path(
        "/gpfs/scratch/shaana01/quantile_regression_root/quantile_bounds_test_combined_batch_14.pt"
        # "/gpfs/scratch/shaana01/quantile_regression_root/quantile_bounds_calib_val_combined_batch_14.pt"
    )

    # Nested sweep: rates ascend so each mask can be built on top of the previous.
    sampling_rates: List[float] = [0.05, 0.10, 0.125, 0.15, 0.20, 0.25]
    base_seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h5_paths = sorted(p for d in calib_dirs for p in d.glob("*.h5"))
    if not h5_paths:
        raise ValueError(f"No HDF5 files found in {calib_dirs}.")

    print(f"Found {len(h5_paths)} volumes across {len(calib_dirs)} directories:", flush=True)
    for d in calib_dirs:
        print(f"  {len(sorted(d.glob('*.h5')))} in {d}", flush=True)
    print(f"Sampling rates (nested): {sampling_rates}", flush=True)

    varnet = load_varnet(varnet_checkpoint, device)
    upper_unet, q_upper = load_quantile_unet(upper_checkpoint, device)
    lower_unet, q_lower = load_quantile_unet(lower_checkpoint, device)
    if not q_upper > 0.5:
        print(f"WARNING: upper checkpoint has quantile={q_upper} (expected > 0.5).", flush=True)
    if not q_lower < 0.5:
        print(f"WARNING: lower checkpoint has quantile={q_lower} (expected < 0.5).", flush=True)

    results: Dict[int, Dict] = {}

    for vol_id, h5_path in enumerate(h5_paths):
        print(f"[{vol_id}] Starting volume: {h5_path.name}", flush=True)
        with h5py.File(h5_path, "r") as hf:
            num_slices = hf["kspace"].shape[0]

        vol_dict: Dict = {"volume_name": h5_path.name}

        for slice_index in range(num_slices):
            kspace, target = load_slice(h5_path, slice_index)
            kspace = kspace.unsqueeze(0).to(device)   # [1, coils, H, W, 2]
            target = target.to(device)                # [H, W]

            # apply_mask-style shape: collapse batch+coil dims -> one column mask
            # broadcast over all coils/slices. Matches quantile module forward.
            mask_shape = (1,) * len(kspace.shape[:-3]) + tuple(kspace.shape[-3:])

            # One mask generator per slice; nesting is chained via previous_mask.
            mask_func = IncrementalVariableDensityMaskFunc(seed=None)
            prev_mask = None
            slice_dict: Dict = {}

            for rate in sampling_rates:
                rate_token = int(round(rate * 100))
                seed = tuple(
                    map(ord, f"{base_seed}:{h5_path.name}:{slice_index}:{rate_token}")
                )

                # Incremental VDS: target_k_fraction sets the new rate; previous_mask
                # forces every already-sampled column to be kept (mask >= prev_mask).
                mask, num_low_frequencies = mask_func(
                    mask_shape,
                    offset=None,
                    seed=seed,
                    target_k_fraction=rate,
                    previous_mask=prev_mask,
                )
                mask = mask.to(device).to(torch.bool)    # [1, 1, 1, W, 1]

                # Direct mask application (no padding clip), as in the quantile
                # module's forward; the +0.0 clears signed zeros.
                masked_kspace = kspace * mask + 0.0

                with torch.no_grad():
                    varnet_recon = varnet(masked_kspace, mask, num_low_frequencies)[0]  # [H, W]

                    target_c, varnet_recon_c = transforms.center_crop_to_smallest(
                        target, varnet_recon
                    )

                    unet_input = varnet_recon_c.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    upper_pred = upper_unet(unet_input)[0, 0]              # [H, W]
                    lower_pred = lower_unet(unet_input)[0, 0]             # [H, W]

                slice_dict[rate] = {
                    "target_rss": target_c.detach().cpu().float(),
                    "mask": mask.detach().squeeze().cpu().bool(),          # [W]
                    "varnet_recon": varnet_recon_c.detach().cpu().float(),
                    "upper_quantile": upper_pred.detach().cpu().float(),
                    "lower_quantile": lower_pred.detach().cpu().float(),
                }

                # Chain: next (higher) rate is built on top of this mask.
                prev_mask = mask

            vol_dict[slice_index] = slice_dict

        results[vol_id] = vol_dict
        print(f"[{vol_id}] Finished volume: {h5_path.name} ({num_slices} slices)", flush=True)

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_pt)
    print(f"Saved: {output_pt}", flush=True)


if __name__ == "__main__":
    main()


# Run from the repo root (the dir that CONTAINS fastmri/ and
# quantile_regression_batches/). The import shim at the top aliases `fastMRI` to
# this repo root, so a plain PYTHONPATH is all that's needed -- no symlink, and
# it works whatever the repo dir is named (e.g. varnet_knee):
#
#   cd /gpfs/home/shaana01/varnet_knee
#   export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
#   python quantile_regression_batches/inference/create_quantile_bounds.py
