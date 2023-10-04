# Decoder Modulation for Indoor Depth Completion

<p align="center">
    <img src="./images/color_1.jpg" width="24%">
    <img src="./images/raw_1.jpg" width="24%">
    <img src="./images/gt_1.jpg" width="24%">
    <img src="./images/pred_1.jpg" width="24%">
 </p>

> **Decoder Modulation for Indoor Depth Completion**<br>
> [Dmitry Senushkin](https://github.com/senush),
> [Ilia Belikov](https://github.com/ferluht),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung AI Center Moscow <br>
> https://arxiv.org/abs/2005.08607

> **Abstract**: *Accurate depth map estimation is an essential step in scene spatial mapping for AR applications and 3D modeling. Current depth sensors provide time-synchronized depth and color images in real-time, but have limited range and suffer from missing and erroneous depth values on transparent or glossy surfaces. We investigate the task of depth completion that aims at improving the accuracy of depth measurements and recovering the missing depth values using additional information from corresponding color images. Surprisingly, we find that a simple baseline model based on modern encoder-decoder architecture for semantic segmentation achieves state-of-the-art accuracy on standard depth completion benchmarks. Then, we show that the accuracy can be further improved by taking into account a mask of missing depth values. The main contributions of our work are two-fold. First, we propose a modified decoder architecture, where features from raw depth and color are modulated by features from the mask via Spatially-Adaptive Denormalization (SPADE). Second, we introduce a new loss function for depth estimation based on direct comparison of log depth prediction with ground truth values. The resulting model outperforms current state-of-the-art by a large margin on the challenging Matterport3D dataset.*

## Installation

To use this code, two ways of installing the necessary libraries are available:

The first one is to use `pip`:

```.bash
pip3 install -r requirements.txt
python setup.py install
```

Or someone can create a conda environment. The original repo provides the `environment.yaml` file for easy setup but it took 12 hours for me when using:
```.bash
conda env create -f environment.yaml
conda activate depth-completion
python setup.py install
```
You can still create a virtual environment by installing the dependencies from the yaml file. 

## Dataset

To train, two datasets have been used. Both are accessible if you follow the guidelines provided here.

### Matterport

The Matterport dataset can be downloaded from the [official website](https://github.com/niessner/Matterport) by signing the Terms of Use agreement form and sending it to the Google email on the website. It is only available for non-commercial academic use only.

A [resized dataset](https://urldefense.com/v3/__https://drive.google.com/drive/folders/1qQ5nMFIdaTgFURBUZYTGTSvn5CELuWus?usp=sharing__;!!JmoZiZGBv3RvKRSx!8ZWSjFGwXccqysnK7jc1ew7KTCJ89j_XZnU8OIUBUarYzJhSjY9UxJjF4WPxBxs3Btsi4ZX1Tng-k8xpmNG4JhmddlIO7A$) is also available for download for everyone. This is the dataset used for training during my internship. Only the `color.zip`, `raw_depth.zip`, and `GT_depth.zip` files are useful for this network, from both train and test folders. \
After extracting the data, restructure the folder so that the data organization looks like this:
```
dataset/
├── 17DRP5sb8fy
│   ├── render_depth
│   │   ├── resize_00ebbf3782c64d74aaf7dd39cd561175_d0_0_mesh_depth.png
│   │   └── ...
│   ├── undistorted_color_images
│   │   ├── resize_00ebbf3782c64d74aaf7dd39cd561175_i0_0.jpg
│   │   └── ...
│   └── undistorted_depth_images
│       ├── resize_00ebbf3782c64d74aaf7dd39cd561175_d0_0.png
│       └── ...
└─── 1LXtFkjw3qL
     └── ...
```
You can now perform inference or training using the `train.txt`, `text.txt`, and `val.txt` files from the `splits` folder. 

### NYUv2

For the NYUv2 dataset, the subset of data that can be used to train the network can be downloaded from the [official website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), it's the `Labeled dataset`. To process the dataset and extract the images and depth maps, you need additional tools. \
My recommendation is to use the code from this [repo](https://github.com/yanyan-li/normal_depth_gt_of_NYU2). Yet, because the depth maps are extracted using the uint8 format, some information is lost during compression. For this reason, please use the `extractRGBD.py` script of this repo that uses the uint32 format instead. 

You can now perform inference or training using the `train_nyu.txt`, `text_nyu.txt`, and `val_nyu.txt` files from the `splits` folder. 

## Pre-processing

Before launching a training or testing script, please ensure that the pre-processing parameters are the ones corresponding to your dataset.  This can be checked and changed in the file `saic_depth_completion/config/dm_lrn.py`. If you use a new dataset for training, please compute the normalization parameters and add them to this file for the pre-processing. 

## Training

For training, multiple scripts are provided depending on the dataset used. For Matterport, use `train_matterport.py`. And for NYUv2, use `train_nyu.py`. 

Depending on whether or not the training is performed with normalized output, the training process changes slightly:

For non-normalized training, the normalization of the output needs to be commented: 

In `/saic_depth_completion/modeling/meta.py`, comment the following:

```bash
# With non normalized output, comment this part
mask_gt = batch["gt_depth"] != 0
batch["gt_depth"][mask_gt] = batch["gt_depth"][mask_gt] - self.depth_mean
batch["gt_depth"][mask_gt] = batch["gt_depth"][mask_gt] / self.depth_std
```

The command to perform the training then depends on the dataset used. All the experiments have been performed with the DM-LRN architecture with the efficientnet b3 backbone. 

```.bash
# for Matterport
python train_matterport.py --default_cfg='DM-LRN' --config_file='../configs/DM-LRN_efficientnet-b3_jenifer.yaml' --weights='<pre-trained weights>' --dataset=<path the matterport dataset> --split=<path to the split folder>

# for NYUv2
python train_nyu.py --default_cfg='DM-LRN' --config_file='../configs/DM-LRN_efficientnet-b3_jenifer.yaml' --weights='<pre-trained weights>' --dataset=<path the matterport dataset> --split=<path to the split folder>
```

## Evaluation
We provide scripts for the evaluation of the results. 

First, you need to modify the dataset on which the validation is performed by changing the `test_dataset` in the file `test_net.py`. Don't forget to also modify the pre-processing to adapt to the test dataset in `saic_depth_completion/config/dm_lrn.py`.

Following instructions perform the evaluation:

```.bash
python test_net.py --default_cfg='DM-LRN' --config_file='../configs/DM-LRN_efficientnet-b3_jenifer.yaml' --weights=<path to the weights> --save_dir=<path to existing folder> --dataset=<path the dataset> --split=<path to the split folder>
```

Some point clouds as well as some visualization images will be saved in the `--save_dir` directory. To generate the point clouds for visualization, open3d needs the intrinsic parameters of the sensor. Modify those parameters in the file `saic_depth_completion/utils/visualize.py` if needed.


## Model ZOO
This repository includes all models mentioned in the original paper. Those models are performing the prediction with a non-normalized output. They only work with the Matterport dataset.

| Backbone | Decoder<br>type   | Encoder<br>input | Training loss |      Link        |  Config |
|----------|-----------|:-----:|:-------------:|:----------------:|:----------:|
| efficientnet-b0 | LRN | RGBD | LogDepthL1loss | [lrn_b0.pth][lrn_b0] | LRN_efficientnet-b0_suzy.yaml |
| efficientnet-b1 | LRN | RGBD | LogDepthL1loss | [lrn_b1.pth][lrn_b1] | LRN_efficientnet-b1_anabel.yaml |
| efficientnet-b2 | LRN | RGBD | LogDepthL1loss | [lrn_b2.pth][lrn_b2] | LRN_efficientnet-b2_irina.yaml |
| efficientnet-b3 | LRN | RGBD | LogDepthL1loss | [lrn_b3.pth][lrn_b3] | LRN_efficientnet-b3_sara.yaml |
| efficientnet-b4 | LRN | RGBD | LogDepthL1loss | [lrn_b4.pth][lrn_b4] | LRN_efficientnet-b4_lena.yaml |
| efficientnet-b4 | LRN | RGBD | BerHu | [lrn_b4_berhu.pth][lrn_b4_berhu] | LRN_efficientnet-b4_helga.yaml |
| efficientnet-b4 | LRN | RGBD+M | LogDepthL1loss | [lrn_b4_mask.pth][lrn_b4_mask] | LRN_efficientnet-b4_simona.yaml |
| efficientnet-b0 | DM-LRN | RGBD | LogDepthL1Loss | [dm-lrn_b0.pth][dm-lrn_b0] | DM_LRN_efficientnet-b0_camila.yaml |
| efficientnet-b1 | DM-LRN | RGBD | LogDepthL1Loss | [dm-lrn_b1.pth][dm-lrn_b1] | DM_LRN_efficientnet-b1_pamela.yaml |
| efficientnet-b2 | DM-LRN | RGBD | LogDepthL1Loss | [dm-lrn_b2.pth][dm-lrn_b2] | DM_LRN_efficientnet-b2_rosaline.yaml |
| efficientnet-b3 | DM-LRN | RGBD | LogDepthL1Loss | [dm-lrn_b3.pth][dm-lrn_b3] | DM_LRN_efficientnet-b3_jenifer.yaml |
| efficientnet-b4 | DM-LRN | RGBD | LogDepthL1Loss | [dm-lrn_b4.pth][dm-lrn_b4] | DM_LRN_efficientnet-b4_pepper.yaml |
| efficientnet-b4 | DM-LRN | RGBD | BerHu | [dm-lrn_b4_berhu.pth][dm-lrn_b4_berhu] | DM_LRN_efficientnet-b4_amelia.yaml |

[lrn_b0]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/lrn_b0.pth
[lrn_b1]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/lrn_b1.pth
[lrn_b2]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/lrn_b2.pth
[lrn_b3]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/lrn_b3.pth
[lrn_b4]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/lrn_b4.pth
[lrn_b4_berhu]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/lrn_b4_berhu.pth
[lrn_b4_mask]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/lrn_b4_mask.pth

[dm-lrn_b0]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/dm-lrn_b0.pth
[dm-lrn_b1]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/dm-lrn_b1.pth
[dm-lrn_b2]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/dm-lrn_b2.pth
[dm-lrn_b3]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/dm-lrn_b3.pth
[dm-lrn_b4]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/dm-lrn_b4.pth
[dm-lrn_b4_berhu]: https://github.com/saic-vul/saic_depth_completion/releases/download/v1.0/dm-lrn_b4_berhu.pth


Some additional trained weights are available in the folder `trained_weights`. Those weights correspond to the training of the network with a normalized depth output. The Herbu loss is used for training.\
They have been included in the repo using `git lfs`.

- Matterport_norm_100.pth : DM-LRN b3 network re-trained on the Matterport dataset with normalized output. Training weights after iteration 100.

- NYUv2_Norm_20.pth : DM-LRN b3 network fine-tuned after the Matterport training with normalized output. The fine-tuning process is done with the NYUv2 dataset and data augmentation (crop + flips). All the parameters of the network are optimized during the fine-tuning. Training weights after iteration 20.

- NYUv2_Norm_50.pth : DM-LRN b3 network fine-tuned after the Matterport training with normalized output. The fine-tuning process is done with the NYUv2 dataset and data augmentation (crop + flips). All the parameters of the network are optimized during the fine-tuning. Training weights after iteration 50.

## License
The code is released under the MPL 2.0 License. MPL is a copyleft license that is easy to comply with. You must make the source code for any of your changes available under MPL, but you can combine the MPL software with proprietary code, as long as you keep the MPL code in separate files.

## Citation
If you find this work useful for your research, please cite our paper:
```
@article{dmidc2020,
  title={Decoder Modulation for Indoor Depth Completion},
  author={Dmitry Senushkin, Ilia Belikov, Anton Konushin},
  journal={arXiv preprint arXiv:2005.08607},
  year={2020}
}
```
