import torch
import numpy as np
import time
import gc

'''
to be run on cpu
This is idea 2
'''

def hoeffding_ucb(risk, n, gamma, B=1.0): # NOTE check implementation of hoeffding_ucb
    """Upper confidence bound on the true risk. risk = empirical risk in [0, B]."""
    return risk + B * np.sqrt(np.log(1 / gamma) / (2 * n))



def psnr_2d(recon, gt, max_pixel_value):
    """
    Standard PSNR between a reconstruction and the fully-sampled ground truth.

    For one slice (2d-image)
    """
    if not (recon.shape == gt.shape and recon.ndim == 2): # should be [H, W]
        raise ValueError(f"shape mismatch: {recon.shape}, {gt.shape}")

    recon = recon.to(torch.float64)
    gt = gt.to(torch.float64)

    mse = ((recon - gt)**2).mean()

    max_pixel_value = torch.as_tensor(max_pixel_value, dtype=torch.float64)

    psnr = 10 * torch.log10(max_pixel_value**2 / mse)
    return psnr


def psnr_3d(recon, gt, max_pixel_value):
    """
    Standard PSNR between a reconstruction and the fully-sampled ground truth.

    For a whole volume (3d) [S, H, W]: MSE pooled over all voxels -> one
    volume-level PSNR.
    """
    if not (recon.shape == gt.shape and recon.ndim == 3): # should be [S, H, W]
        raise ValueError(f"shape mismatch: {recon.shape}, {gt.shape}")

    recon = recon.to(torch.float64)
    gt = gt.to(torch.float64)

    mse = ((recon - gt)**2).mean()

    max_pixel_value = torch.as_tensor(max_pixel_value, dtype=torch.float64)

    psnr = 10 * torch.log10(max_pixel_value**2 / mse)
    return psnr



def main():


    
    sampling_rates = [0.05, 0.10, 0.20, 0.25, 0.35, 0.50]
    # Inputs: the two create_quantile_bounds outputs (one per split). Update to
    # your actual paths if they differ.
    data_path_calibration = "/gpfs/scratch/shaana01/conformal-mri-reconstruction-logs/quantile_bounds_calibration_config_14.pt"
    data_path_test = "/gpfs/scratch/shaana01/conformal-mri-reconstruction-logs/quantile_bounds_test_config_14.pt"
    # save_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/calibrated_lambda_1.pt" # NOTE lambdas for 2d psnr
    # save_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/calibrated_lambda_2.pt"# NOTE lambdas for 3d psnr
    # NOTE: grid itself is no longer saved (per-slice (score, psnr) tuples are kept
    # in memory only for the lambda sweep). We save score_by_rate instead.
    # save_grid_path = "/gpfs/scratch/shaana01/quantile_regression_root/score_grid_batch_14_2.pt"
    save_score_by_rate_path = "/gpfs/scratch/shaana01/quantile_regression_root/score_by_rate_batch_14_2.pt"
    save_psnr_by_rate_path = "/gpfs/scratch/shaana01/quantile_regression_root/psnr_by_rate_batch_14_2.pt"
    save_rejections_by_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/rejections_by_lambda_batch_14_2.pt"
    save_psnr_by_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/psnr_by_lambda_batch_14_2.pt"
    save_coverage_by_min_psnr_path = "/gpfs/scratch/shaana01/quantile_regression_root/coverage_by_min_psnr_batch_14_2.pt"
    save_chosen_lambdas_path = "/gpfs/scratch/shaana01/quantile_regression_root/chosen_lambdas_by_min_psnr_batch_14_2.pt"
    save_score_psnr_pairs_path = "/gpfs/scratch/shaana01/quantile_regression_root/score_psnr_pairs_batch_14.pt"


    print(f"Loading calibration data from {data_path_calibration} ...", flush=True)
    _t_load = time.time()
    data_calib = torch.load(data_path_calibration, map_location=torch.device('cpu'))
    print(f"Loaded calibration data in {time.time() - _t_load:.1f}s", flush=True)
    # NOTE: test data is loaded later (TEST SPLIT section), AFTER data_calib is
    # freed, so calibration and test data are never both in RAM at once.

    num_vols = 0
    # count num_vols
    for vol_id in data_calib.keys():
        if not isinstance(vol_id, int):
            continue
        num_vols += 1
    print(f"num_vols = {num_vols}", flush=True)
    print()

    # finding max pixel value across all volumes and slices. NOTE: check what max pixel value should actually be for psnr calculation.
    max_pixel_value = -1
    for vol_id in data_calib.keys():
        if not isinstance(vol_id, int):
            continue
        for slc_id in data_calib[vol_id].keys():
            if not isinstance(slc_id, int):
                continue
            # NOTE(max_val): PSNR data-range should be the ground-truth peak (standard
            # fastMRI convention), and target_rss is rate-independent (unlike varnet_recon,
            # which varies with the mask). Old line kept below.
            # max_pixel_value = max(max_pixel_value, torch.max(data_calib[vol_id][slc_id][sampling_rates[-1]]['varnet_recon']).item())
            max_pixel_value = max(max_pixel_value, torch.max(data_calib[vol_id][slc_id][sampling_rates[-1]]['target_rss']).item())
    print(f"max_pixel_value = {max_pixel_value}", flush=True)
    print()

    # 2d grid, now PER-SLICE: one row per (volume, slice), one column per sampling
    # rate (columns follow `sampling_rates`, col 0 = min rate, last col = max rate).
    # Each cell is a (score, psnr) tuple:
    #   score = 1 / (mean per-pixel quantile width over THIS slice * width_scale)
    #           -> larger = tighter interval = better
    #   psnr  = per-slice 2d PSNR (varnet recon vs RSS target)
    # row_keys[i] = (vol_id, slc_id) for grid row i (same order).
    #
    # MRI intensities are tiny, so widths are reported in x10^4 units (width_scale)
    # for readability. Keep the SAME scale downstream or the units won't match.
    # score_by_rate / psnr_by_rate lists are built in row order, so
    # score_by_rate[rate][i], psnr_by_rate[rate][i], and grid[i] are the same slice.
    width_scale = 1e4
    grid = []              # grid[i][j] = (score, psnr) for slice i at sampling_rates[j]
    row_keys = []          # (vol_id, slc_id) per grid row, same order
    psnr_by_rate = {rate: [] for rate in sampling_rates}
    score_by_rate = {rate: [] for rate in sampling_rates}
    score_psnr_pairs = []  # flat list of (score, psnr), one per (slice, rate) -- for a scatter plot
    for vol_id in data_calib.keys():
        if not isinstance(vol_id, int):
            continue
        slc_ids = sorted(s for s in data_calib[vol_id].keys() if isinstance(s, int))
        for slc_id in slc_ids:
            row_keys.append((vol_id, slc_id))
            row = []
            for rate in sampling_rates:
                entry = data_calib[vol_id][slc_id][rate]

                # score: mean per-pixel quantile width over THIS slice (2d).
                width = (entry['upper_quantile'] - entry['lower_quantile']) * width_scale
                score = width.sum().item() / width.numel()
                grid_score = 1 / score   # store 1/score so that larger = better

                # psnr: same (slice, rate), 2d.
                recon = entry['varnet_recon']
                gt = entry['target_rss']
                psnr_val = psnr_2d(recon, gt, max_pixel_value).item()

                row.append((grid_score, psnr_val))
                score_by_rate[rate].append(grid_score)
                psnr_by_rate[rate].append(psnr_val)
                score_psnr_pairs.append((grid_score, psnr_val))
            grid.append(row)

    print(f"grid: {len(grid)} slices x {len(sampling_rates)} rates", flush=True)
    print(f"score/psnr_by_rate: {len(sampling_rates)} rates x {len(row_keys)} slices", flush=True)
    print()

    # score spread at the min and max sampling rate (from score_by_rate).
    first_rate, last_rate = sampling_rates[0], sampling_rates[-1]
    fc, lc = score_by_rate[first_rate], score_by_rate[last_rate]
    print(f"min rate ({first_rate}): "
          f"max={max(fc):.6f} min={min(fc):.6f} avg={sum(fc) / len(fc):.6f}", flush=True)
    print(f"max rate ({last_rate}): "
          f"max={max(lc):.6f} min={min(lc):.6f} avg={sum(lc) / len(lc):.6f}", flush=True)

    # ---- save score_by_rate + psnr_by_rate + score_psnr_pairs ----
    # (grid is kept in memory only -- used by the lambda sweep below, not saved.)

    # score_by_rate.pt: dict sampling_rate(float) -> list[float] of the per-slice
    # score (1/width, larger = better), one per slice, in grid row order.
    torch.save(score_by_rate, save_score_by_rate_path)
    print(f"Saved score_by_rate -> {save_score_by_rate_path}", flush=True)

    print()

    # psnr_by_rate.pt: dict sampling_rate(float) -> list[float] of the per-slice
    # 2d PSNR, one per slice, in the same row order as score_by_rate / grid.
    torch.save(psnr_by_rate, save_psnr_by_rate_path)
    print(f"Saved psnr_by_rate -> {save_psnr_by_rate_path}", flush=True)

    print()

    # score_psnr_pairs.pt: a flat python list of (score, psnr) tuples, one per
    # (slice, rate) pair (len = num_slices * num_rates). score = 1/width (larger =
    # better); psnr = per-slice 2d PSNR. For a scatter plot.
    torch.save(score_psnr_pairs, save_score_psnr_pairs_path)
    print(f"Saved score_psnr_pairs -> {save_score_psnr_pairs_path}", flush=True)

    print()
    print('---------------------------------------------------------------', flush=True)
    print()

    # Sweep lambda (a score budget, same units as the grid scores). For each lambda,
    # each SLICE (grid row) picks the column whose score is the MAXIMUM one still
    # <= lambda (the highest usable sampling rate under the budget); its PSNR is read
    # straight from the grid tuple. A slice with no score <= lambda has no valid pick
    # and is counted as rejected/skipped for that lambda.
    #
    # For each min-PSNR threshold in set_min_psnr we track, per lambda, the fraction
    # of ALL slices whose chosen pick meets that PSNR (coverage), and record the
    # smallest lambda whose coverage reaches delta.






    # ************************************************************************
    set_min_psnr = [30.0, 40.0, 50.0]
    delta = 0.90
    # gamma = 0.15
    starting_lambda = 15
    ending_lambda = 3
    num_steps = 15
    lambdas = np.linspace(starting_lambda, ending_lambda, num_steps)  # descending
    # ************************************************************************


    total_num_slices = len(grid)   # one candidate sample (a chosen slice,rate pick) per slice

    # A "sample" at a given lambda = one CHOSEN (slice, sampling_rate) pick. Each slice
    # yields one sample if some rate has score <= lambda, else it is rejected.
    rejections_by_lambda = {}   # lambda -> # samples rejected (slices with no valid pick)
    psnr_by_lambda = {}         # lambda -> list of chosen-sample PSNRs (len <= total_num_slices)
    coverage_by_min_psnr = {mp: {} for mp in set_min_psnr}          # mp -> {lambda: coverage in [0,1]}
    chosen_lambda_for_min_psnr = {mp: None for mp in set_min_psnr}  # mp -> smallest lambda hitting delta

    for lam in lambdas:
        lam_key = round(float(lam), 3)
        psnr_list = []          # chosen-sample PSNR for kept samples at this lambda
        skipped = 0
        for row in grid:
            # pick the column (rate) with the MAX score still <= lambda.
            best_idx = -1
            for idx, (score, _p) in enumerate(row):
                if score <= lam and (best_idx == -1 or row[best_idx][0] < score):
                    best_idx = idx
            if best_idx == -1:
                skipped += 1
                continue
            psnr_list.append(row[best_idx][1])   

        rejections_by_lambda[lam_key] = skipped
        psnr_by_lambda[lam_key] = psnr_list

        # coverage per min-PSNR threshold (denominator = total_num_slices, so
        # rejected samples count as NOT meeting the threshold).
        for mp in set_min_psnr:
            num_ok = sum(1 for p in psnr_list if p >= mp)
            pct = num_ok / total_num_slices
            coverage_by_min_psnr[mp][lam_key] = pct
            # smallest lambda whose coverage reaches delta.
            if pct >= delta and (
                chosen_lambda_for_min_psnr[mp] is None
                or lam < chosen_lambda_for_min_psnr[mp]
            ):
                chosen_lambda_for_min_psnr[mp] = float(lam)

        if psnr_list:
            print(f"lambda={lam:.3f}: n={len(psnr_list)} skipped={skipped} "
                  f"psnr[max={max(psnr_list):.2f} min={min(psnr_list):.2f} "
                  f"avg={sum(psnr_list) / len(psnr_list):.2f}]", flush=True)
        else:
            print(f"lambda={lam:.3f}: n=0 skipped={skipped} (no sample had a score <= lambda)", flush=True)

    print()
    print(f"Smallest lambda reaching {100 * delta:.0f}% coverage per min-PSNR:", flush=True)
    for mp in set_min_psnr:
        print(f"  min_psnr={mp:.1f} -> lambda={chosen_lambda_for_min_psnr[mp]}", flush=True)
    print()

    # ---- save lambda-sweep bookkeeping ----
    # rejections_by_lambda.pt: dict lambda(float) -> int, # samples rejected (slices
    # with no rate whose score <= lambda) at that lambda. For rejection-count vs lambda plots.
    torch.save(rejections_by_lambda, save_rejections_by_lambda_path)
    print(f"Saved rejections_by_lambda -> {save_rejections_by_lambda_path}", flush=True)

    # psnr_by_lambda.pt: dict lambda(float) -> list[float] of chosen-sample PSNRs
    # (one per KEPT sample; len <= total_num_slices, shrinks as samples get rejected).
    # For a PSNR box-and-whisker vs lambda.
    torch.save(psnr_by_lambda, save_psnr_by_lambda_path)
    print(f"Saved psnr_by_lambda -> {save_psnr_by_lambda_path}", flush=True)

    # coverage_by_min_psnr.pt: dict min_psnr(float) -> {lambda(float): coverage in
    # [0,1]} = fraction of ALL slices whose chosen sample meets that PSNR at that
    # lambda (rejected samples count as not meeting). For coverage-vs-lambda curves.
    torch.save(coverage_by_min_psnr, save_coverage_by_min_psnr_path)
    print(f"Saved coverage_by_min_psnr -> {save_coverage_by_min_psnr_path}", flush=True)

    # chosen_lambdas_by_min_psnr.pt: dict min_psnr(float) -> smallest lambda whose
    # coverage reached delta (None if never reached).
    torch.save(chosen_lambda_for_min_psnr, save_chosen_lambdas_path)
    print(f"Saved chosen_lambda_for_min_psnr -> {save_chosen_lambdas_path}", flush=True)

    print()
    print('---------------------------------------------------------------', flush=True)
    print()

    # ============================================================
    # TEST SPLIT
    # ============================================================
    # Free the calibration data first -- everything above kept only floats (grid,
    # score/psnr dicts, chosen lambdas), so nothing below needs data_calib. This
    # reclaims its RAM before the (large) test data is loaded, so the two splits
    # are never both resident at once.
    del data_calib
    gc.collect()

    print(f"Loading test data from {data_path_test} ...", flush=True)
    _t_load = time.time()
    data_test = torch.load(data_path_test, map_location=torch.device('cpu'))
    print(f"Loaded test data in {time.time() - _t_load:.1f}s", flush=True)

    # TODO(test): apply the chosen lambda(s) to data_test and report realized
    # coverage / PSNR on the held-out test split.



if __name__ == "__main__":
      main()


