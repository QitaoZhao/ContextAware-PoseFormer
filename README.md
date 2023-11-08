# ContextAware-PoseFormer

This repo is the official implementation for **A Single 2D Pose with Context is Worth Hundreds for 3D Human Pose Estimation**. The paper has been accepted to [NeurIPS 2023](https://nips.cc/).

[arXiv](https://arxiv.org/pdf/2311.03312.pdf) / [project page](https://qitaozhao.github.io/ContextAware-PoseFormer)

### News

[2023.11.06] Our paper on [arXiv](https://arxiv.org/pdf/2311.03312.pdf) has been released. We are preparing for the video narration; some parts of the code still need cleaning. 

## Introduction

**Abstract.** The dominant paradigm in 3D human pose estimation that lifts a 2D pose sequence to 3D heavily relies on long-term temporal clues (i.e., using a daunting number of video frames) for improved accuracy, which incurs performance saturation, intractable computation and the non-causal problem. This can be attributed to their inherent inability to perceive spatial context as plain 2D joint coordinates carry no visual cues. To address this issue, we propose a straightforward yet powerful solution: leveraging the *readily available* intermediate visual representations produced by off-the-shelf (pre-trained) 2D pose detectors -- no finetuning on the 3D task is even needed. The key observation is that, while the pose detector learns to localize 2D joints, such representations (e.g., feature maps) implicitly encode the joint-centric spatial context thanks to the regional operations in backbone networks. We design a simple baseline named **Context-Aware PoseFormer** to showcase its effectiveness. *Without access to any temporal information*, the proposed method significantly outperforms its context-agnostic counterpart, PoseFormer, and other state-of-the-art methods using up to *hundreds of* video frames regarding both speed and precision. 

![framework](./images/framework.png)

### Dataset Preparation

1. Please refer to [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) to set up RGB images from the Human3.6M dataset. We include this repo in our project for your convenience. All RGB images should be put under `code_root/H36M-Toolbox/images/`. 

    **Note**: Only RGB images take around 200GB of disk space, while intermediate files take even more space. Carefully remove these intermediate files after each step if you do not have sufficient disk space.

2. Download `data_2d_h36m_cpn_ft_h36m_dbb.npz` from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md) and put it under `code_root/H36M-Toolbox/data/`. This file contains pre-processed CPN-detected keypoints.

3. We modify `generate_labels_h36m.py` from `generate_labels.py` provided by [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox). Run `generate_labels_h36m.py` to generate labels (`h36m_train.pkl` and `h36m_validation.pkl`) for training and testing. It may take a while.

4. Your directory should look like this if you correctly follow the previous steps.

```
code_root/ 
├── README.md
├── ContextPose/
└── H36M-Toolbox/
    ├── ...
    ├── generate_labels_h36m.py
    ├── h36m_train.pkl
    ├── h36m_validation.pkl
    ├── data/
    	└── data_2d_h36m_cpn_ft_h36m_dbb.npz
    └── images/
        ├── s_01_act_02_subact_01_ca_01/
            ├── s_01_act_02_subact_01_ca_01_000001.jpg
            ├── ...
            └── s_01_act_02_subact_01_ca_01_001384.jpg
        ├── s_01_act_02_subact_01_ca_02/
        ├── ...
        └── s_11_act_16_subact_02_ca_04/
```

### Environment

The code is developed and tested under the following environment.

- Python 3.8.10
- PyTorch 1.11.0
- CUDA 11.3

````
cd ContextPose
conda create --name conposeformer --file conda-requirements.txt
conda activate conposeformer
pip install -r requirements.txt
````

### Train

It's time to train your model! Our framework supports multiple backbone 2D joint detectors. For now, we take [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) as an example.

1. Move `h36m_train.pkl` and `h36m_validation.pkl` from `code_root/H36M-Toolbox/` to `code_root/ContextPose/data/`
2. Download (COCO) pretrained weights for HRNet-32 from https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA and place it under `code_root/ContextPose/data/pretrained/coco/`
3. Your directory should look like this. Now, you are ready for training!

```
code_root/ 
├── README.md
├── H36M-Toolbox/
└── ContextPose/
    ├── ...
    ├── train.py
    ├── experiments/
    ├── data/
    	├── h36m_train.pkl
    	├── h36m_validation.pkl
    	└── pretrained/
    		└── coco/
    			├── README.MD
    			└── pose_hrnet_w32_256x192.pth
    └── mvn/
```

You can train **Context-aware PoseFormer** on a single GPU with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config experiments/human36m/train/human36m_vol_softmax_single.yaml --logdir ./logs
```

If you want to train on multiple (e.g., 4) GPUS, simply do the following:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=2345 train.py --config experiments/human36m/train/human36m_vol_softmax_single.yaml --logdir ./logs
```

### Test

The checkpoint corresponding to the log above is available [here](https://drive.google.com/file/d/1nh8BLCyEFaoRGhb_sFmwvU5xWJLlATi4/view?usp=sharin). You should get 41.3mm MPJPE using this checkpoint. Place the trained model (`best_epoch.bin`) under `code_root/ContextPose/checkpoint/`, and run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=2345 train.py --config experiments/human36m/train/human36m_vol_softmax_single.yaml --logdir ./logs --eval
```

## Cite Our Work

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{
    zhao2023contextaware,
    title={A Single 2D Pose with Context is Worth Hundreds for 3D Human Pose Estimation},
    author={Zhao, Qitao and Zheng, Ce and Liu, Mengyuan and Chen, Chen},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

## Acknowledgment

Our codes are largely based on [ContextPose](https://github.com/ShirleyMaxx/ContextPose-PyTorch-release) and [PoseFormer](https://github.com/zczcwh/PoseFormer). We follow [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) for data preparation. Many thanks to the authors!

