import torch
import numpy as np

# from calibrate_lambda import psnr_lower_bound_2d, psnr_2d
from conformal.calibrate_lambda import psnr_lower_bound_2d, psnr_2d

'''
to be run on cpu
'''


def main():

    # ---- settings ----
    vol_id = 0
    sampling_rates = [0.05, 0.10, 0.125, 0.15, 0.20, 0.25]
    data_path = "/gpfs/scratch/shaana01/quantile_regression_root/quantile_bounds_calib_batch_14.pt"
    lambda_path = "/gpfs/scratch/shaana01/quantile_regression_root/calibrated_lambda_1.pt"

    # ---- load data and the calibrated (vanilla) lambda ----
    data = torch.load(data_path, map_location=torch.device('cpu'))
    calibrated = torch.load(lambda_path, map_location=torch.device('cpu'))
    lambda_ = calibrated["chosen_lambda_vanilla"]
    print(f"Using lambda (vanilla) = {lambda_}")

    # ---- find max pixel value across all volumes and slices (same as calibrate_lambda.py) ----
    max_pixel_value = -1
    for v in data.keys():
        if not isinstance(v, int):
            continue
        for slc_id in data[v].keys():
            if not isinstance(slc_id, int):
                continue
            # NOTE(max_val): use ground-truth peak (target_rss) as PSNR data-range --
            # standard fastMRI convention and rate-independent. Old line kept below.
            # max_pixel_value = max(max_pixel_value, torch.max(data[v][slc_id][sampling_rates[-1]]['varnet_recon']).item())
            max_pixel_value = max(max_pixel_value, torch.max(data[v][slc_id][sampling_rates[-1]]['target_rss']).item())
    print(f"max_pixel_value = {max_pixel_value}")

    # ---- worst-case and ground-truth PSNR per sampling rate for this volume (averaged over slices) ----
    print(f"vol_id = {vol_id}")
    for t_ in sampling_rates:
        psnr_calc = []
        psnr_gt = []
        for slc_id in data[vol_id].keys():
            if not isinstance(slc_id, int):
                continue

            recon = data[vol_id][slc_id][t_]['varnet_recon']
            target_rss = data[vol_id][slc_id][t_]['target_rss']
            lower_quantile = data[vol_id][slc_id][t_]['lower_quantile']
            upper_quantile = data[vol_id][slc_id][t_]['upper_quantile']

            lb = recon - lambda_ * (recon - lower_quantile)
            ub = recon + lambda_ * (upper_quantile - recon)

            psnr_low = psnr_lower_bound_2d(recon, lb, ub, max_pixel_value)
            psnr = psnr_2d(recon, target_rss, max_pixel_value)
            psnr_calc.append(psnr_low.item())
            psnr_gt.append(psnr.item())

        worst_case_psnr = np.mean(psnr_calc)
        actual_psnr = np.mean(psnr_gt)
        print(f"sampling_rate: {t_}, worst_case_psnr: {worst_case_psnr}, gt_psnr: {actual_psnr}")


if __name__ == "__main__":
    main()
