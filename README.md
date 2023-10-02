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

The forst one is to use `pip`:

```.bash
pip3 install -r requirements.txt
python setup.py install
```

Or someone can create a conda environment. The original repo is providing the `environment.yaml` file for easy setup but it was not working for me. Using `Anaconda` is still recommanded but a manual installation of the libraries is need by going through the `enviromment.yaml` file.

## Dataset



## Pre-processing

Before launching a training or testing script, please ensure that the pre-processing parameters are the one corresponding to you dataset.  This can be checked and changed in the file `saic_depth_completion/config/dm_lrn.py`. If you use a new dataset for training, please complute the normalization parameters and add them in this file for the pre-processing. 

## Training

For training, multiple scripts are provided depending on the dataset used. For Matterport, use `train_matterport.py`. And for NYUv2, use `train_nyu.py`. 

Depending on wether or not the training in performed with normalized output, the training process changes slightly:

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
We provide scripts for evaluation od the results. 

First, you need to modify the dataset on which the validation is performed by changing the `test_dataset` in the file `test_net.py`. Don't forget to also modify the pre-processing to adapt to the test dataset in `saic_depth_completion/config/dm_lrn.py`.

Following instructions performs evaluation:

```.bash
python test_net.py --default_cfg='DM-LRN' --config_file='../configs/DM-LRN_efficientnet-b3_jenifer.yaml' --weights=<path to the weights> --save_dir=<path to existing folder> --dataset=<path the dataset> --split=<path to the split folder>
```

Some point clouds as well as some visualization images will be saved in the `--save_dir` directory. To generate the point clouds for visualization, open3d needs the intrinsic parameters of the sensor. Modify those parameters in the file `saic_depth_completion/utils/visualize.py` if needed.


## Model ZOO
This repository includes all models mentioned in original paper. 

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

## SUPP

We provide a code for training on [Matterport3D](https://github.com/patrickwu2/Depth-Completion/blob/master/doc/data.md). Download Matterpord3D dataset and reorder your root folder as follows:
```bash
ROOT/
  ├── data/
  └── splits/
        ├── train.txt
        ├── val.txt
        └── test.txt 
```


and `data` directory is should be configured in [this order](https://github.com/patrickwu2/Depth-Completion/blob/master/doc/data.md). Be sure that ROOT path in [matterport.py](https://github.sec.samsung.net/d-senushkin/saic_depth_completion_public/blob/master/saic_depth_completion/data/datasets/matterport.py) is valid. 
Now you can start training with the following command:



## License
The code is released under the MPL 2.0 License. MPL is a copyleft license that is easy to comply with. You must make the source code for any of your changes available under MPL, but you can combine the MPL software with proprietary code, as long as you keep the MPL code in separate files.

## Citation
If you find this work is useful for your research, please cite our paper:
```
@article{dmidc2020,
  title={Decoder Modulation for Indoor Depth Completion},
  author={Dmitry Senushkin, Ilia Belikov, Anton Konushin},
  journal={arXiv preprint arXiv:2005.08607},
  year={2020}
}
```
