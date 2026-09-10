import torch
import numpy as np
import time

'''
to be run on cpu
This is idea 2
'''

def hoeffding_ucb(risk, n, gamma, B=1.0): # NOTE check implementation of hoeffding_ucb
    """Upper confidence bound on the true risk. risk = empirical risk in [0, B]."""
    return risk + B * np.sqrt(np.log(1 / gamma) / (2 * n))



# def psnr_2d(recon, gt, max_pixel_value):
#     """
#     Standard PSNR between a reconstruction and the fully-sampled ground truth.

#     For one slice (2d-image)
#     """
#     if not (recon.shape == gt.shape and recon.ndim == 2): # should be [H, W]
#         raise ValueError(f"shape mismatch: {recon.shape}, {gt.shape}")

#     recon = recon.to(torch.float64)
#     gt = gt.to(torch.float64)

#     mse = ((recon - gt)**2).mean()

#     max_pixel_value = torch.as_tensor(max_pixel_value, dtype=torch.float64)

#     psnr = 10 * torch.log10(max_pixel_value**2 / mse)
#     return psnr


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

    # delta = 0.10
    # gamma = 0.15
    # start_lambda = 2.0
    # end_lambda = 0.5
    # lambda_step = 30

    
    sampling_rates = [0.05, 0.10, 0.125, 0.15, 0.20, 0.25]
    data_path =  "/gpfs/scratch/shaana01/quantile_regression_root/quantile_bounds_calib_val_combined_batch_14.pt"
    # save_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/calibrated_lambda_1.pt" # NOTE lambdas for 2d psnr
    # save_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/calibrated_lambda_2.pt"# NOTE lambdas for 3d psnr
    save_grid_path = "/gpfs/scratch/shaana01/quantile_regression_root/score_grid_batch_14_2.pt"
    save_psnr_by_rate_path = "/gpfs/scratch/shaana01/quantile_regression_root/psnr_by_rate_batch_14_2.pt"
    save_rejections_by_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/rejections_by_lambda_batch_14_2.pt"
    save_psnr_by_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/psnr_by_lambda_batch_14_2.pt"
    save_score_psnr_pairs_path = "/gpfs/scratch/shaana01/quantile_regression_root/score_psnr_pairs_batch_14.pt"


    print(f"Loading data from {data_path} ...", flush=True)
    _t_load = time.time()
    data = torch.load(data_path, map_location=torch.device('cpu'))
    print(f"Loaded data in {time.time() - _t_load:.1f}s", flush=True)

    
    num_vols = 0
    # count num_vols
    for vol_id in data.keys():
        if not isinstance(vol_id, int):
            continue
        num_vols += 1
    print(f"num_vols = {num_vols}", flush=True)
    print()

    # finding max pixel value across all volumes and slices. NOTE: check what max pixel value should actually be for psnr calculation.
    max_pixel_value = -1
    for vol_id in data.keys():
        if not isinstance(vol_id, int):
            continue
        for slc_id in data[vol_id].keys():
            if not isinstance(slc_id, int):
                continue
            # NOTE(max_val): PSNR data-range should be the ground-truth peak (standard
            # fastMRI convention), and target_rss is rate-independent (unlike varnet_recon,
            # which varies with the mask). Old line kept below.
            # max_pixel_value = max(max_pixel_value, torch.max(data[vol_id][slc_id][sampling_rates[-1]]['varnet_recon']).item())
            max_pixel_value = max(max_pixel_value, torch.max(data[vol_id][slc_id][sampling_rates[-1]]['target_rss']).item())
    print(f"max_pixel_value = {max_pixel_value}", flush=True)
    print()

    # 2d grid of average quantile widths: one row per volume, one column per
    # sampling rate (columns follow `sampling_rates`, so col 0 = min rate, last
    # col = max rate). For a (volume, rate) pair the score is the total quantile
    # width (upper - lower) summed over every pixel of every slice, divided by
    # the total pixel count of the volume -- i.e. the mean per-pixel width.
    #
    # MRI intensities are tiny, so we report in x10^4 units for readability.
    # Applied once here at the source, so grid / first_col / last_col all inherit
    # it. Keep the SAME scale downstream (lambda calibration) or the units won't match.
    # Single pass over (volume, rate): compute BOTH the score and the volume-level
    # 3d PSNR here, since we already have the slices in hand -- no second traversal.
    # psnr_by_rate lists are built in vol_ids order (so psnr_by_rate[rate][i] and
    # grid[i] are the same volume).
    width_scale = 1e4
    grid = []
    vol_ids = []   # vol_id for each grid row, same order, so we can fetch slices later
    psnr_by_rate = {rate: [] for rate in sampling_rates}
    score_psnr_pairs = []   # flat list of (score, psnr), one per (volume, rate) -- for a scatter plot
    for vol_id in data.keys():
        if not isinstance(vol_id, int):
            continue
        vol_ids.append(vol_id)
        slc_ids = sorted(s for s in data[vol_id].keys() if isinstance(s, int))
        row = []
        for rate in sampling_rates:
            # score: mean per-pixel quantile width over the whole volume.
            width_sum = 0.0
            pixel_count = 0
            for slc_id in slc_ids:
                entry = data[vol_id][slc_id][rate]
                width = (entry['upper_quantile'] - entry['lower_quantile']) * width_scale
                width_sum += width.sum().item()
                pixel_count += width.numel()
            score = width_sum / pixel_count
            grid_score = 1 / score   # store 1/score so that larger = better
            row.append(grid_score)

            # psnr: same (volume, rate), reusing the slices we already gathered.
            recon = torch.stack([data[vol_id][s][rate]['varnet_recon'] for s in slc_ids])
            gt = torch.stack([data[vol_id][s][rate]['target_rss'] for s in slc_ids])
            psnr_val = psnr_3d(recon, gt, max_pixel_value).item()
            psnr_by_rate[rate].append(psnr_val)

            score_psnr_pairs.append((grid_score, psnr_val))
        grid.append(row)

    print(f"grid: {len(grid)} volumes x {len(sampling_rates)} rates", flush=True)
    print(f"psnr_by_rate: {len(psnr_by_rate)} rates x {len(vol_ids)} volumes", flush=True)
    print()

    # first column = min sampling rate, last column = max sampling rate.
    first_col = [row[0] for row in grid]
    last_col = [row[-1] for row in grid]

    print(f"min rate ({sampling_rates[0]}): "
          f"max={max(first_col):.6f} min={min(first_col):.6f} "
          f"avg={sum(first_col) / len(first_col):.6f}", flush=True)
    print(f"max rate ({sampling_rates[-1]}): "
          f"max={max(last_col):.6f} min={min(last_col):.6f} "
          f"avg={sum(last_col) / len(last_col):.6f}", flush=True)
    
    # ---- save grid + psnr_by_rate ----

    # score_grid.pt: a python list of lists, len = num volumes. grid[i] is one
    # volume; grid[i][j] is its score at sampling_rates[j] (col 0 = 0.05 ... col
    # -1 = 0.25). score = 1 / (mean per-pixel quantile width * width_scale), so
    # LARGER = tighter interval = better. Row order matches psnr_by_rate lists.
    torch.save(grid, save_grid_path)
    print(f"Saved grid -> {save_grid_path}", flush=True)

    print()

    # ---- psnr_by_rate ----
    # (psnr_by_rate was computed above in the same pass as the grid.)
    # psnr_by_rate.pt: a python dict, sampling_rate(float) -> list[float] of the
    # volume-level 3d PSNR for each volume at that rate. Every list has len =
    # num volumes, in the same volume order as the grid rows.
    torch.save(psnr_by_rate, save_psnr_by_rate_path)
    print(f"Saved psnr_by_rate -> {save_psnr_by_rate_path}", flush=True)

    print()

    # ---- score_psnr_pairs ----
    # score_psnr_pairs.pt: a flat python list of (score, psnr) tuples, one per
    # (volume, rate) pair (len = num_volumes * num_rates). score = grid value
    # (1/width, larger = better); psnr = volume-level 3d PSNR. For a scatter plot.
    torch.save(score_psnr_pairs, save_score_psnr_pairs_path)
    print(f"Saved score_psnr_pairs -> {save_score_psnr_pairs_path}", flush=True)

    print()
    print('---------------------------------------------------------------', flush=True)
    print()

    # Sweep lambda (a width budget, same units as grid) from high to low. For each
    # lambda, each volume picks the row score that is the MAXIMUM one still <= lambda
    # -- i.e. the smallest quantile width that fits the budget -- and we score that
    # (volume, sampling_rate) pick with a volume-level PSNR (varnet recon vs RSS).
    # A volume with no score <= lambda (even its tightest width exceeds the budget)
    # has no valid pick and is skipped for that lambda.
    starting_lambda = 15
    ending_lambda = 3
    num_steps = 15
    lambdas = np.linspace(starting_lambda, ending_lambda, num_steps)  # descending

    rejections_by_lambda = {}   # lambda -> int count of rejected (skipped) volumes
    psnr_by_lambda = {}         # lambda -> list of psnr values (one per kept volume)

    for lam in lambdas:
        psnr_list = []
        skipped = 0
        for row, vol_id in zip(grid, vol_ids):
            
            best_idx = -1
            for idx, score in enumerate(row):
                if score <= lam and (best_idx == -1 or row[best_idx] < score):
                    best_idx = idx
            if best_idx == -1:
                skipped += 1
                continue

            rate = sampling_rates[best_idx]
            slc_ids = sorted(s for s in data[vol_id].keys() if isinstance(s, int))
            recon = torch.stack([data[vol_id][s][rate]['varnet_recon'] for s in slc_ids])
            gt = torch.stack([data[vol_id][s][rate]['target_rss'] for s in slc_ids])
            psnr_list.append(psnr_3d(recon, gt, max_pixel_value).item())

        # bookkeeping (reuses values already computed above -- no extra work).
        lam_key = round(float(lam), 3)
        rejections_by_lambda[lam_key] = skipped
        psnr_by_lambda[lam_key] = psnr_list

        if psnr_list:
            print(f"lambda={lam:.3f}: n={len(psnr_list)} skipped={skipped} "
                  f"max={max(psnr_list):.4f} min={min(psnr_list):.4f} "
                  f"avg={sum(psnr_list) / len(psnr_list):.4f}", flush=True)
            print()
        else:
            print(f"lambda={lam:.3f}: n=0 skipped={skipped} (no volume had a score <= lambda)",
                  flush=True)
            print()

    # ---- save lambda-sweep bookkeeping ----
    # rejections_by_lambda.pt: a python dict, lambda(float) -> int, the number of
    # volumes rejected (no score <= lambda) at that lambda. Keys are in descending
    # lambda order. For plotting rejection count vs decreasing lambda.
    torch.save(rejections_by_lambda, save_rejections_by_lambda_path)
    print(f"Saved rejections_by_lambda -> {save_rejections_by_lambda_path}", flush=True)

    # psnr_by_lambda.pt: a python dict, lambda(float) -> list[float] of the
    # volume-level 3d PSNR for each KEPT volume at that lambda (list length varies
    # with lambda as volumes get rejected). For a box-and-whisker of PSNR spread
    # vs decreasing lambda.
    torch.save(psnr_by_lambda, save_psnr_by_lambda_path)
    print(f"Saved psnr_by_lambda -> {save_psnr_by_lambda_path}", flush=True)
   


    


if __name__ == "__main__":
      main()


