import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import open3d as o3d

def save_images(color, raw_depth, mask, gt, pred, regressed_post_pred, save_dir, idx):
    color       = np.array(color.cpu().permute(1, 2, 0))
    raw_depth   = np.array(raw_depth.cpu())
    gt          = gt.cpu()
    mask        = mask.cpu()
    pred        = np.array(pred.detach().cpu())

    fig = plt.figure()
    plt.imshow(mask[0], cmap='RdBu_r', vmin=mask[0].min(), vmax=mask[0].max())
    fig.savefig(os.path.join(save_dir, "mask_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    vmin = min(regressed_post_pred.min(), gt.min())
    vmax = max(regressed_post_pred.max(), gt.max())

    fig = plt.figure()
    plt.imshow((color - color.min()) / (color.max() - color.min()))
    fig.savefig(os.path.join(save_dir, "color_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(raw_depth[0], cmap='RdBu_r', vmin=raw_depth[0].min(), vmax=raw_depth[0].max())
    fig.savefig(os.path.join(save_dir, "raw_depth_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(regressed_post_pred, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.savefig(os.path.join(save_dir, "pred_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(gt[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.savefig(os.path.join(save_dir, "gt_{}.png".format(idx)), dpi=fig.dpi)
    plt.close(fig)

    return

def save_point_cloud_to_ply(filename, depth, color):
    # Convert numpy arrays to open3d Image objects
    depth_o3d = o3d.geometry.Image(depth)
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(color).astype(np.uint8))

    # Create an RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,  # Adjust based on your depth scale
        depth_trunc=float('inf'),
        convert_rgb_to_intensity=False
    )

    # Create a point cloud from the RGBD image using intrinsic parameters
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        depth.shape[1], depth.shape[0],
        912.62, 911.689,   # fx, fy
        depth.shape[1] // 2, depth.shape[0] // 2
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, pinhole_camera_intrinsic
    )

    # Save point cloud as PLY
    o3d.io.write_point_cloud(filename, pcd)

def save_pc_o3d(raw_color, color, raw_depth, mask, gt, pred, pred_regressed, save_dir, idx):

    mask        = np.array(mask.cpu(), dtype=int)[0]
    raw_color   = np.array(raw_color.cpu().permute(1, 2, 0))*255
    gt          = np.array(gt.cpu())[0]
    pred        = np.array(pred.detach().cpu())[0]

    hole_filling = np.copy(gt)
    hole_filling[mask  == 0] = pred_regressed[mask == 0]

    save_point_cloud_to_ply(os.path.join(save_dir, "pc_pred_{}.ply".format(idx)), pred_regressed, raw_color)
    save_point_cloud_to_ply(os.path.join(save_dir, "pc_gt_{}.ply".format(idx)), gt, raw_color)
    save_point_cloud_to_ply(os.path.join(save_dir, "pc_hole_filling_{}.ply".format(idx)), hole_filling, raw_color)
    return