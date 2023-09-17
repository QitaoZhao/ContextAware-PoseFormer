# ContextAware-PoseFormer

This repo is the official implementation for **Context-aware PoseFormer: Single Image Beats Hundreds for 3D Human Pose Estimation**. 

### Dataset Preparation

1. Please refer to [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) to set up RGB images from Human3.6M dataset. We include this repo in our project for your convenience. All RGB images should be put under `code_root/H36M-Toolbox/images/`.
2. Download `data_2d_h36m_cpn_ft_h36m_dbb.npz` from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md) and put it under `code_root/H36M-Toolbox/data/`. This file contains pre-processed CPN-detected keypoints.
3. We modify `generate_labels_h36m.py` from `generate_labels.py` provided by [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox). Run `generate_labels_h36m.py` to generate labels (`h36m_train.pkl` and `h36m_validation.pkl`) for training and testing. It may take a while.
4. Your directory should look like this if you correctly follow previous steps.

```
code_root/ 
├── README.md
├── ContextPose-PyTorch-release/
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

### Training

It's time to train your model! Our framework supports multiple backbone 2D joint detectors. For now, we take [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) as an example.

1. Move `h36m_train.pkl` and `h36m_validation.pkl` from `code_root/H36M-Toolbox/` to `code_root/ContextPose-PyTorch-release/data/`
2. Download (COCO) pretrained weights for HRNet-32 from https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA and place it under `code_root/ContextPose-PyTorch-release/data/pretrained/coco/`
3. Your directory should look like this. Now, you are ready for training!

```
code_root/ 
├── README.md
├── H36M-Toolbox/
└── ContextPose-PyTorch-release/
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

You can train Context-aware PoseFormer on a single GPU with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config experiments/human36m/train/human36m_vol_softmax_single.yaml --logdir ./logs
```

If you want to train on multiple (e.g., 4) GPUS, simply do the following:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=2345 train.py --config experiments/human36m/train/human36m_vol_softmax_single.yaml --logdir ./logs
```





