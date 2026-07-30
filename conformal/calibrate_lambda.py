import torch
import numpy as np
import time

'''
to be run on cpu
'''

def hoeffding_ucb(risk, n, gamma, B=1.0): # NOTE check implementation of hoeffding_ucb
    """Upper confidence bound on the true risk. risk = empirical risk in [0, B]."""
    return risk + B * np.sqrt(np.log(1 / gamma) / (2 * n))


def psnr_lower_bound_2d(recon, lb, ub, max_pixel_value):
    """
    Worst-case (lower-bound) PSNR of a reconstruction against a pixel-wise
    interval set C_lambda, computed without access to the ground truth.

    For one slice (2d-image) and to be run on CPU.
    """
    if not (recon.shape == lb.shape == ub.shape and recon.ndim == 2): # should be [H, W]
        raise ValueError(f"shape mismatch: {recon.shape}, {lb.shape}, {ub.shape}")

    recon = recon.to(torch.float64)
    lb = lb.to(torch.float64)
    ub = ub.to(torch.float64)

    per_pixel_max_sq = torch.maximum((recon - lb)**2, (ub - recon)**2)
    mse_max = per_pixel_max_sq.mean() 

    max_pixel_value = torch.as_tensor(max_pixel_value, dtype=torch.float64)

    psnr_low = 10 * torch.log10(max_pixel_value**2 / mse_max)
    return psnr_low



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



def psnr_lower_bound_3d(recon, lb, ub, max_pixel_value):
    """
    Worst-case (lower-bound) PSNR of a reconstruction against a pixel-wise
    interval set C_lambda, computed without access to the ground truth.

    For a whole volume (3d) [S, H, W]: the max per-voxel squared error is pooled
    over ALL voxels (every slice, every pixel) into one MSE, giving a single
    volume-level worst-case PSNR. To be run on CPU.
    """
    if not (recon.shape == lb.shape == ub.shape and recon.ndim == 3): # should be [S, H, W]
        raise ValueError(f"shape mismatch: {recon.shape}, {lb.shape}, {ub.shape}")

    recon = recon.to(torch.float64)
    lb = lb.to(torch.float64)
    ub = ub.to(torch.float64)

    per_voxel_max_sq = torch.maximum((recon - lb)**2, (ub - recon)**2)
    mse_max = per_voxel_max_sq.mean()

    max_pixel_value = torch.as_tensor(max_pixel_value, dtype=torch.float64)

    psnr_low = 10 * torch.log10(max_pixel_value**2 / mse_max)
    return psnr_low



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

    delta = 0.10
    gamma = 0.15
    start_lambda = 2.0
    end_lambda = 0.5
    lambda_step = 30

    
    sampling_rates = [0.05, 0.10, 0.125, 0.15, 0.20, 0.25]
    data_path =  "/gpfs/scratch/shaana01/quantile_regression_root/quantile_bounds_calib_val_combined_batch_14.pt"
    # save_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/calibrated_lambda_1.pt" # NOTE lambdas for 2d psnr
    save_lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/calibrated_lambda_2.pt"# NOTE lambdas for 3d psnr


    print(f"Loading data from {data_path} ...", flush=True)
    _t_load = time.time()
    data = torch.load(data_path, map_location=torch.device('cpu'))
    print(f"Loaded data in {time.time() - _t_load:.1f}s", flush=True)

    lambdas = np.linspace(start_lambda, end_lambda, lambda_step)
    print(f"Sweeping {len(lambdas)} lambdas from {start_lambda} to {end_lambda} "
          f"(delta={delta}, gamma={gamma})", flush=True)

    num_vols = 0
    # count num_vols
    for vol_id in data.keys():
        if not isinstance(vol_id, int):
            continue
        num_vols += 1
    print(f"num_vols = {num_vols}", flush=True)

    # finding max pixel value across all volumes and slices. NOTE: check max pixel value, it is currently same for both psnr calculations
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

    chosen_lambda_vanilla = 100.0
    chosen_lambda_hoeffding = 100.0

    for i, lambda_ in enumerate(lambdas):
        t0 = time.time()
        print(f"[{i + 1}/{len(lambdas)}] evaluating lambda = {lambda_:.4f} ...", flush=True)
        risk_count = 0

        for vol_id in data.keys():
            if not isinstance(vol_id, int):
                continue

            for t_ in sampling_rates:

                # NOTE(2d->3d): old per-slice loop that averaged per-slice PSNRs
                # (mean of dB). Replaced by the volume-stacked 3d version below.
                # psnr_calc = []
                # psnr_gt = []
                # for slc_id in data[vol_id].keys():
                #     if not isinstance(slc_id, int):
                #         continue
                #
                #     recon = data[vol_id][slc_id][t_]['varnet_recon']
                #     target_rss = data[vol_id][slc_id][t_]['target_rss']
                #     lower_quantile = data[vol_id][slc_id][t_]['lower_quantile']
                #     upper_quantile = data[vol_id][slc_id][t_]['upper_quantile']
                #
                #     lb = recon - lambda_ * (recon - lower_quantile)
                #     ub = recon + lambda_ * (upper_quantile - recon)
                #
                #     psnr_low = psnr_lower_bound_2d(recon, lb, ub, max_pixel_value)
                #     psnr = psnr_2d(recon, target_rss, max_pixel_value)
                #     psnr_calc.append(psnr_low.item())
                #     psnr_gt.append(psnr.item())
                #
                # worst_case_psnr = np.mean(psnr_calc)
                # actual_psnr = np.mean(psnr_gt)

                slice_ids = [s for s in data[vol_id].keys() if isinstance(s, int)]

                recon = torch.stack([data[vol_id][s][t_]['varnet_recon'] for s in slice_ids])
                target_rss = torch.stack([data[vol_id][s][t_]['target_rss'] for s in slice_ids])
                lower_quantile = torch.stack([data[vol_id][s][t_]['lower_quantile'] for s in slice_ids])
                upper_quantile = torch.stack([data[vol_id][s][t_]['upper_quantile'] for s in slice_ids])

                lb = recon - lambda_ * (recon - lower_quantile)
                ub = recon + lambda_ * (upper_quantile - recon)

                worst_case_psnr = psnr_lower_bound_3d(recon, lb, ub, max_pixel_value).item()
                actual_psnr = psnr_3d(recon, target_rss, max_pixel_value).item()

                if actual_psnr < worst_case_psnr: # Risk violated
                    risk_count += 1
                    break


        risk = risk_count / num_vols
        risk_hoeffding = hoeffding_ucb(risk, num_vols, gamma)
        if risk <= delta:
            chosen_lambda_vanilla = lambda_
        if risk_hoeffding <= delta:
            chosen_lambda_hoeffding = lambda_

        print(f"lambda: {lambda_}, risk: {risk}, risk_hoeffding: {risk_hoeffding}, "
              f"risk_count: {risk_count}/{num_vols}  ({time.time() - t0:.1f}s)", flush=True)
        print(flush=True)

        if risk > delta and risk_hoeffding > delta:
            print(f"Both risks exceed delta={delta}; stopping sweep at lambda={lambda_:.4f}", flush=True)
            break

    print(f"Chosen lambda (vanilla): {chosen_lambda_vanilla}", flush=True)
    print(f"Chosen lambda (Hoeffding): {chosen_lambda_hoeffding}", flush=True)

    calibrated = {
        "chosen_lambda_vanilla": float(chosen_lambda_vanilla),
        "chosen_lambda_hoeffding": float(chosen_lambda_hoeffding),
    }
    torch.save(calibrated, save_lambda_path)
    print(f"Saved calibrated lambdas to {save_lambda_path}", flush=True)



if __name__ == "__main__":
      main()

# NOTE
'''
1. Early-break assumes risk is monotonic in lambda. It increases as lambda shrinks in expectation, but calibration-set noise could make it non-monotone; a spike could stop the sweep
early. Ans: its fine

2. Does gamma need to be split across lambdas? Does this also need a multiple hypothesis style correction? Ans: No. its fixed sequence testing and not LTT


- max value
- psnr 2d vs 3d
- hoeffding is fine
- 
'''
