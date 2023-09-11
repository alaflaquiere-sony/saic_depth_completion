import os
import time
import datetime
import torch
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from saic_depth_completion.utils import visualize


def inference(
        model, test_loaders, metrics, save_dir="", logger=None
):

    model.eval()
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
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

            start = time.time()
            pred = model(batch)
            end = time.time()
            inference_times.append(end - start)

            with torch.no_grad():
                post_pred = model.postprocess(pred)
                if save_dir and idx<50:
                    # B = batch["color"].shape[0]
                    # for it in range(B):
                    #     fig = visualize.figure(
                    #         batch["color"][it], batch["raw_depth"][it],
                    #         batch["mask"][it], batch["gt_depth"][it],
                    #         post_pred[it], close=True
                    #     )
                    #     fig.savefig(
                    #         os.path.join(save_dir, "result_{}.png".format(idx)), dpi=fig.dpi
                    #     )

                    #     idx += 1
                    B = batch["color"].shape[0]
                    for it in range(B):
                        visualize.save_images(
                            batch["color"][it], batch["raw_depth"][it],
                            batch["mask"][it], batch["gt_depth"][it],
                            post_pred[it], save_dir, idx
                            )

                        idx += 1
                metrics_meter.update(post_pred, batch["gt_depth"], batch["mask"])
        print('Mean inference time:', np.mean(inference_times[3:]))

        state = "Inference: subset -- {} | ".format(subset)
        logger.info(state + metrics_meter.suffix)