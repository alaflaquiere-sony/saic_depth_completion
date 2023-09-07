import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def figure(color, raw_depth, mask, gt, pred, close=False):
    fig, axes = plt.subplots(3, 2, figsize=(7, 10))

    color       = color.cpu().permute(1, 2, 0)
    raw_depth   = raw_depth.cpu()
    mask        = mask.cpu()
    gt          = gt.cpu()
    pred        = pred.detach().cpu()

    vmin = min(gt.min(), pred.min(), raw_depth.min())
    vmax = max(gt.max(), pred.max(), raw_depth.max())

    axes[0, 0].set_title('RGB')
    axes[0, 0].imshow((color - color.min()) / (color.max() - color.min()) )

    axes[0, 1].set_title('raw_depth')
    img = axes[0, 1].imshow(raw_depth[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=axes[0, 1])

    axes[1, 0].set_title('mask')
    axes[1, 0].imshow(mask[0])

    axes[1, 1].set_title('gt')
    img = axes[1, 1].imshow(gt[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=axes[1, 1])

    axes[2, 1].set_title('pred')
    img = axes[2, 1].imshow(pred[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=axes[2, 1])
    if close: plt.close(fig)
    return fig

def save_images(color, raw_depth, mask, gt, pred, save_dir, idx):
    color       = np.array(color.cpu().permute(1, 2, 0))
    raw_depth   = np.array(raw_depth.cpu())
    gt          = gt.cpu()
    mask        = mask.cpu()
    pred        = np.array(pred.detach().cpu())

    fig = plt.figure()
    plt.imshow(mask[0], cmap='RdBu_r', vmin=mask[0].min(), vmax=mask[0].max())
    fig.savefig(os.path.join(save_dir, "mask_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    vmin = min(raw_depth.min(), pred.min(), gt.min())
    vmax = max(raw_depth.max(), pred.max(), gt.max())

    fig = plt.figure()
    plt.imshow((color - color.min()) / (color.max() - color.min()))
    fig.savefig(os.path.join(save_dir, "color_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(raw_depth[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.savefig(os.path.join(save_dir, "raw_depth_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(pred[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.savefig(os.path.join(save_dir, "pred_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(gt[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.savefig(os.path.join(save_dir, "gt_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    # cv2.imwrite(os.path.join(save_dir, "color_{}.png".format(idx)), color)
    # cv2.imwrite(os.path.join(save_dir, "raw_depth_{}.png".format(idx)), raw_depth)
    # cv2.imwrite(os.path.join(save_dir, "pred_{}.png".format(idx)), pred)
    return

def save_1_image(color, raw_depth, mask, gt, pred, save_dir, idx):
    color       = np.array(color.cpu().permute(1, 2, 0))
    raw_depth   = np.array(raw_depth.cpu())
    gt          = gt.cpu()
    mask        = mask.cpu()
    pred        = np.array(pred.detach().cpu())
    print(color.shape, cv2.cvtColor(np.array(gt.permute(1, 2, 0)), cv2.COLOR_GRAY2RGB).shape, cv2.cvtColor(pred.transpose(1, 2, 0), cv2.COLOR_GRAY2RGB).shape)
    concat = cv2.hconcat([(color - color.min()) / (color.max() - color.min()), cv2.cvtColor(np.array(gt.permute(1, 2, 0)), cv2.COLOR_GRAY2RGB)]) #, cv2.cvtColor(pred.transpose(1, 2, 0), cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join(save_dir, "concat_result_{}.png".format(idx)), concat)
    return