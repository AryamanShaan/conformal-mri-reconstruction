from __future__ import annotations

# Variant of varnet_ssim_score.py for a VarNet trained with the fastMRI
# leaderboard script (fastmri_examples/.../varnet_knee_leaderboard.py), which
# trains a Lightning `VarNetModule` and saves Lightning `.ckpt` files.
#
# ONLY the checkpoint loading + forward/shape differ from varnet_ssim_score.py:
#   * checkpoint is a Lightning VarNetModule .ckpt: weights live under
#     state_dict["varnet.*"], architecture in ckpt["hyper_parameters"]. We load
#     them into a bare fastmri.models.VarNet (no Lightning import needed -> works
#     regardless of the env's pytorch_lightning version).
#   * fastMRI VarNet.forward(masked_kspace, mask, num_low_frequencies) returns
#     [B, H, W] (rss over coils) -- NOT [B, 1, H, W] like TGVN's VarNetImage --
#     so num_low_frequencies is passed and the SSIM shapes are handled for 3-D.
# Reports two SSIMs per rate:
#   * skimage SSIM with per-slice (max - min) data_range, and
#   * "evalpy" SSIM = a verbatim copy of fastMRI utilities/evaluation.py `ssim`,
#     fed the way the VarNet repo feeds it (data_range = volume max attrs["max"]).

import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Imports use the fastMRI-repo namespace (fastMRI.fastmri.*), matching the other
# scripts in this repo (e.g. varnet_knee_leaderboard.py). The custom VDS mask
# lives in the fork's subsample module here (not tgvn.masks).
# from fastmri.data import transforms
from common import functions as transforms
# from fastmri.data.mri_data import et_query
from common.mri_header import et_query
# from fastmri.models import VarNet
from e2evarnet.VarNet import E2EVarNet as VarNet
# from fastmri.data.subsample import RandomVariableDensityMaskFunc
from common.undersampling_patterns import RandomVariableDensityMaskFunc


def load_varnet(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a leaderboard-trained fastMRI VarNet from a Lightning .ckpt.

    The Lightning VarNetModule stores the network under 'varnet.' in state_dict
    and its constructor args in ckpt['hyper_parameters']. We rebuild the bare
    fastmri.models.VarNet from those and load the stripped state_dict, avoiding
    any pytorch_lightning import (robust to the env's PL version).
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

    # Lightning stores the net under 'varnet.' -> strip that prefix.
    vsd = {k[len("varnet."):]: v for k, v in state.items() if k.startswith("varnet.")}
    if not vsd:  # already a bare varnet state_dict
        vsd = state
    missing, unexpected = model.load_state_dict(vsd, strict=False)
    print(f"load_varnet: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model.eval()
    model.to(device)
    return model


def load_slice(
    h5_path: Path,
    slice_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    with h5py.File(h5_path, "r") as hf:
        kspace = hf["kspace"][slice_index]
        target = hf["reconstruction_rss"][slice_index]
        attrs = dict(hf.attrs)

        et_root = etree.fromstring(hf["ismrmrd_header"][()])
        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (
            int(et_query(et_root, enc + ["x"])),
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["z"])),
        )
        lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        enc_limits_center = int(et_query(et_root, lims + ["center"]))
        enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max

        attrs["padding_left"] = padding_left
        attrs["padding_right"] = padding_right

    return transforms.to_tensor(kspace), transforms.to_tensor(target), attrs


def make_masked_kspace(
    kspace: torch.Tensor,
    attrs: Dict,
    sampling_fraction: float,
    seed: Tuple[int, ...],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    # min == max pins the mask to exactly this sampling fraction; padding applies
    # the acquisition-support clipping. Returns num_low_frequencies (needed by the
    # fastMRI VarNet's sensitivity model for the ACS region).
    mask_func = RandomVariableDensityMaskFunc(
        min_k_fraction=sampling_fraction,
        max_k_fraction=sampling_fraction,
    )

    masked_kspace, mask, num_low_frequencies = transforms.apply_mask(
        kspace,
        mask_func,
        seed=seed,
        padding=(attrs["padding_left"], attrs["padding_right"]),
    )

    return (
        masked_kspace.unsqueeze(0).to(device),          # [1, coils, H, W, 2]
        mask.unsqueeze(0).to(device).to(torch.bool),    # [1, 1, 1, W, 1]
        num_low_frequencies,
    )


def compute_ssim_skimage(target_2d: np.ndarray, recon_2d: np.ndarray) -> float:
    data_range = target_2d.max() - target_2d.min()
    if data_range == 0:
        data_range = 1.0
    return float(structural_similarity(target_2d, recon_2d, data_range=data_range))


# Verbatim copy of fastMRI utilities/evaluation.py `ssim` -- the exact function
# the VarNet repo uses for its val/ssim and test-set SSIM. It is fed here exactly
# as that repo feeds it: ssim(target[None, ...], output[None, ...], maxval=attrs["max"]).
def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]


def main() -> None:
    # --- inputs (EDIT THESE) ---
    data_dir = Path("/gpfs/scratch/shaana01/knee_fastmri_cp/multicoil_test")
    # Leaderboard-trained VarNet: a Lightning .ckpt under <root>/checkpoints/
    varnet_checkpoint = Path(
        "/gpfs/scratch/shaana01/varnet_knee_root_dir/run_1/checkpoints/epochepoch=33-val_lossvalidation_loss=0.1117.ckpt"
    )
    output_pt = Path(
        "/gpfs/scratch/shaana01/varnet_ssim_results/leaderboard_run_1_ssim_scores.pt"
    )
    sampling_fractions = [0.10, 0.15, 0.20, 0.25, 0.30]
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- optional: SSIM-spread exports (for box-and-whisker plots later) ---
    # The `results` table saved at the end keeps only *averaged* SSIM (one number
    # per rate), which hides the distribution. These two optional dumps preserve
    # the spread so a downstream Jupyter nb can boxplot it. Both use the "evalpy"
    # SSIM (data_range = volume max attrs["max"]) -- the fastMRI leaderboard
    # convention, i.e. column (b) of the printed table below. Each is gated by its
    # own flag, so you only spend cluster disk on the .pt you actually want.
    #
    # (1) per-SLICE spread: { sampling_fraction: [ssim for EVERY slice at that
    #     rate, pooled across all volumes] } -- finest-grained distribution.
    #     Needs in-loop accumulation, so we init the dict here (None if disabled).
    save_slice_ssim_spread = True
    slice_ssim_spread_pt = Path(
        "/gpfs/scratch/shaana01/varnet_ssim_results/leaderboard_run_1_slice_ssim_spread.pt"
    )
    slice_ssim_by_rate = (
        {sf: [] for sf in sampling_fractions} if save_slice_ssim_spread else None
    )
    #
    # (2) per-VOLUME spread: { sampling_fraction: [one volume-averaged ssim per
    #     volume at that rate] } -- coarser distribution. Built from per_rate_ev
    #     at the very end (that list already holds the volume averages), so there
    #     is nothing extra to accumulate inside the loop.
    save_volume_ssim_spread = True
    volume_ssim_spread_pt = Path(
        "/gpfs/scratch/shaana01/varnet_ssim_results/leaderboard_run_1_volume_ssim_spread.pt"
    )

    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        raise ValueError(f"No HDF5 files found in {data_dir}.")

    print(f"Found {len(h5_paths)} volumes in {data_dir}", flush=True)
    print(f"Sampling fractions: {sampling_fractions}", flush=True)

    varnet = load_varnet(varnet_checkpoint, device)

    per_rate_sk = [[] for _ in sampling_fractions]
    per_rate_ev = [[] for _ in sampling_fractions]

    for h5_path in h5_paths:
        print(f"Starting volume: {h5_path.name}", flush=True)
        with h5py.File(h5_path, "r") as hf:
            num_slices = hf["kspace"].shape[0]

        for rate_idx, sampling_fraction in enumerate(sampling_fractions):
            slice_ssims_sk = []
            slice_ssims_ev = []

            for slice_index in range(num_slices):
                kspace, target, attrs = load_slice(h5_path, slice_index)
                max_value = float(attrs["max"])

                rate_token = int(round(sampling_fraction * 100))
                mask_seed = tuple(
                    map(ord, f"{seed}:{h5_path.name}:{slice_index}:{rate_token}")
                )

                masked_kspace, mask, num_low_frequencies = make_masked_kspace(
                    kspace=kspace,
                    attrs=attrs,
                    sampling_fraction=sampling_fraction,
                    seed=mask_seed,
                    device=device,
                )

                with torch.no_grad():
                    # fastMRI VarNet returns [B, H, W] (rss over coils).
                    out = varnet(masked_kspace, mask, num_low_frequencies)

                    # target -> [B, H, W]; crop both to the smallest H/W.
                    target_3d = target.to(device).unsqueeze(0)
                    target_c, out_c = transforms.center_crop_to_smallest(target_3d, out)

                # 2-D cropped images: target = ground truth, recon = prediction.
                t2 = target_c.squeeze().cpu().numpy()
                r2 = out_c.squeeze().cpu().numpy()

                # (a) skimage SSIM, per-slice (max - min) data_range.
                ssim_sk = compute_ssim_skimage(t2, r2)
                # (b) evaluate.py SSIM, fed exactly as the VarNet repo feeds it:
                #     ssim(target[None, ...], output[None, ...], maxval=attrs["max"]).
                ssim_ev = float(ssim(t2[None, ...], r2[None, ...], maxval=max_value))

                slice_ssims_sk.append(ssim_sk)
                slice_ssims_ev.append(ssim_ev)

                # per-slice spread export: pool this slice's SSIM under its rate.
                if save_slice_ssim_spread:
                    slice_ssim_by_rate[sampling_fraction].append(ssim_ev)

            per_rate_sk[rate_idx].append(float(np.mean(slice_ssims_sk)))
            per_rate_ev[rate_idx].append(float(np.mean(slice_ssims_ev)))

        print(f"Finished volume: {h5_path.name}", flush=True)

    # --- print and save ---
    results = []
    print("\n" + "=" * 62, flush=True)
    print(f"{'rate':>6} | {'skimage SSIM':>14} | {'evalpy SSIM':>14}", flush=True)
    print("-" * 62, flush=True)
    for sampling_fraction, vols_sk, vols_ev in zip(sampling_fractions, per_rate_sk, per_rate_ev):
        avg_sk = float(np.mean(vols_sk))
        avg_ev = float(np.mean(vols_ev))
        results.append((sampling_fraction, avg_sk, avg_ev))
        print(f"{sampling_fraction:>6.2f} | {avg_sk:>14.6f} | {avg_ev:>14.6f}", flush=True)
    print("=" * 62 + "\n", flush=True)

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    # each entry: (sampling_fraction, avg_skimage_ssim, avg_evalpy_ssim)
    torch.save(results, output_pt)
    print(f"Saved: {output_pt}", flush=True)

    # optional SSIM-spread dumps (each guarded by its flag set at the top of main)
    if save_slice_ssim_spread:
        slice_ssim_spread_pt.parent.mkdir(parents=True, exist_ok=True)
        # dict: { sampling_fraction: [per-slice evalpy SSIM, pooled over all volumes] }
        torch.save(slice_ssim_by_rate, slice_ssim_spread_pt)
        print(f"Saved slice-SSIM spread: {slice_ssim_spread_pt}", flush=True)

    if save_volume_ssim_spread:
        volume_ssim_spread_pt.parent.mkdir(parents=True, exist_ok=True)
        # dict: { sampling_fraction: [per-volume-averaged evalpy SSIM] }
        # per_rate_ev[i] is already the list of volume averages for rate i.
        volume_ssim_by_rate = {
            sf: vols for sf, vols in zip(sampling_fractions, per_rate_ev)
        }
        torch.save(volume_ssim_by_rate, volume_ssim_spread_pt)
        print(f"Saved volume-SSIM spread: {volume_ssim_spread_pt}", flush=True)


if __name__ == "__main__":
    main()


# Run from the conformal-mri-reconstruction repo root so the `common.*` /
# `e2evarnet.*` packages resolve (put the repo root on PYTHONPATH):
# export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# python e2evarnet/inference/varnet_ssim_scores.py
