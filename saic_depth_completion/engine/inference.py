import os
import time
import datetime
import torch
from tqdm import tqdm
import numpy as np
import cv2
import warnings
from datetime import timedelta
import time

import matplotlib.pyplot as plt

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from saic_depth_completion.utils import visualize


def polynomial_regression(source_values, reference_values, degree=2):
    """Perform polynomial regression to find a mapping from source to reference values."""
    coeffs = np.polyfit(source_values, reference_values, degree)
    return coeffs

def regression(source_img, reference_img, mask=None, degree=2):
    """Match the source image to the reference image using polynomial regression."""
    
    # If a mask is provided, consider only the pixels where mask == 1
    if mask is not None:
        source_values = source_img[mask == 1]
        reference_values = reference_img[mask == 1]
    else:
        source_values = source_img.ravel()
        reference_values = reference_img.ravel()

    # Sort the source and reference values
    sorted_indices = np.argsort(source_values)
    sorted_source = source_values[sorted_indices]
    sorted_reference = reference_values[sorted_indices]

    # Get the polynomial coefficients
    coeffs = polynomial_regression(sorted_source, sorted_reference, degree)

    # Apply the polynomial mapping to the source image
    matched_img = np.polyval(coeffs, source_img).clip(0, None).astype(source_img.dtype)
    
    return matched_img


def compute_patchwise_coefficients(source_img, reference_img, mask, patch_size=32, degree=2, min_valid_pixels=100):
    """Compute polynomial coefficients patch-wise using a mask."""
    height, width = source_img.shape
    coeffs_list = []

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            source_patch = source_img[i:i+patch_size, j:j+patch_size]
            reference_patch = reference_img[i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]

            # Consider only valid (masked) pixels
            valid_source = source_patch[mask_patch == 1]
            valid_reference = reference_patch[mask_patch == 1]

            # If there are enough valid pixels, compute and store coefficients for this patch
            if len(valid_source) > min_valid_pixels:
                coeffs = polynomial_regression(valid_source, valid_reference, degree)
                coeffs_list.append(coeffs)

    return np.array(coeffs_list)


def eliminate_outliers(coeffs_list):
    """Eliminate outlier coefficients using Median Absolute Deviation (MAD)."""
    median = np.median(coeffs_list, axis=0)
    mad = np.median(np.abs(coeffs_list - median), axis=0)
    # Define a threshold for outliers
    threshold = 2.5
    non_outliers = np.abs(coeffs_list - median) < threshold * mad
    # Keep only those coefficients which are not outliers in any of the polynomial's terms
    filtered_coeffs = coeffs_list[np.all(non_outliers, axis=1)]
    return filtered_coeffs


def patchwise_polynomial_regression(source_img, reference_img, mask_img ,patch_size=32, degree=2):
    """Match the histogram of the source image to that of the reference image using polynomial regression and patches."""
    # Compute patch-wise coefficients
    coeffs_list = compute_patchwise_coefficients(source_img, reference_img, mask_img, patch_size=patch_size, degree=degree)

    # Eliminate outliers
    filtered_coeffs = eliminate_outliers(coeffs_list)

    # Compute the mean of the remaining coefficients
    mean_coeffs = np.mean(filtered_coeffs, axis=0)

    # Apply the polynomial mapping to the source image
    output_img = np.polyval(mean_coeffs, source_img).clip(0, source_img.max()).astype(source_img.dtype)

    return output_img


def inverse_normalization(source_img):
    # Realsense coeffs
    post_process_image = source_img * 0.12419240655170437 + 0.4320115620826656

    # Matterport coeffs
    # post_process_image = source_img * 1.4279 + 2.1489

    # NYUv2 coeffs
    # post_process_image = source_img * 0.81951 + 2.79619
    return post_process_image


def inference(
        model, test_loaders, metrics, save_dir="", logger=None
):

    model.eval()
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    warnings.simplefilter('ignore', np.RankWarning)
    for subset, loader in test_loaders.items():
        idx = 0
        logger.info(
            "Inference: subset -- {}. Total number of batches: {}.".format(subset, len(loader))
        )

        metrics_meter.reset()
        inference_times = []
        # loop over dataset
        for batch in tqdm(loader):
            batch = model.preprocess(batch)
            pred = model(batch)

            with torch.no_grad():
                post_pred = model.postprocess(pred)

                if save_dir and idx<20:
                    B = batch["color"].shape[0]
                    for it in range(B):
                        # Perform domain adaptation with polynomial regression

                        # Patch wise regression
                        # depth_post_pred = patchwise_polynomial_regression(np.array(post_pred[it].cpu())[0], np.array(batch["gt_depth"][it].cpu())[0], np.array(batch["mask"][it].cpu())[0], degree=2)
                        # Normal regression
                        # depth_post_pred = regression(np.array(post_pred[it].cpu())[0], np.array(batch["gt_depth"][it].cpu())[0], mask=np.array(batch["mask"][it].cpu())[0], degree=2)

                        # Post processing for the normalized network (that predicts normalized output)
                        depth_post_pred = inverse_normalization(np.array(post_pred[it].cpu())[0])

                        visualize.save_pc_o3d(
                            batch["raw_color"][it], batch["color"][it], batch["raw_depth"][it],
                            batch["mask"][it], batch["gt_depth"][it],
                            post_pred[it], depth_post_pred, save_dir, idx
                            )

                        visualize.save_images(
                            batch["color"][it], batch["raw_depth"][it],
                            batch["mask"][it], batch["gt_depth"][it],
                            post_pred[it], depth_post_pred, save_dir, idx
                            )
                        idx += 1
                metrics_meter.update(post_pred, batch["gt_depth"], batch["mask"])
        print('Mean inference time:', np.mean(inference_times[3:]))

        state = "Inference: subset -- {} | ".format(subset)
        logger.info(state + metrics_meter.suffix)
