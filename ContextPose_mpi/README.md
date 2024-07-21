## MPI-INF-3DHP

We heavily borrowed code from [P-STMO](https://github.com/paTRICK-swk/P-STMO) to train and evaluate our model on MPI-INF-3DHP.

**Note:** We did not use Deformable Context Extraction for this dataset as our input is ground truth 2D keypoint.

### Dataset Preparation

1. Download and pre-process data by running (which may take a while to complete):

    ~~~shell
    bash dataset/process_data.sh
    ~~~

    This handles (1) Data download, (2) Extracting labels, and (3) Processing raw videos. 

2. Download (COCO) pre-trained weights for HRNet-32/HRNet-48 from https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA and place it under `dataset/pretrained/`.


3. Your `dataset` directory should look like this if you follow the previous steps correctly.

```bash
dataset/ 
├── process_data.sh
├── data_train_3dhp.npz
├── data_test_3dhp.npz
├── data_util/
└── mpi_inf_3dhp/
    ├── ...
    └── images/
└── mpi_inf_3dhp_test_set/
    ├── ...
    └── images/
└── pretrained/
    ├── pose_hrnet_w32_256x192.pth
    └── pose_hrnet_w48_256x192.pth
```

### Train

Use the following command to train our HRNet-32 model:

```
python run_3dhp.py -f 1 -b 160 --train 1 --lr 0.0007 -lrd 0.97 --backbone hrnet_32
```

Similarly, for HRNet-48, run the following command:

```
python run_3dhp.py -f 1 -b 160 --train 1 --lr 0.0007 -lrd 0.97 --backbone hrnet_48
```

### Evaluation

A simple evaluation can be done by running:

```
python run_3dhp.py -f 1 -b 160 --train 0 --reload 1 --backbone hrnet_32
```

Likewise, run this for the HRNet-48 model:

```
python run_3dhp.py -f 1 -b 160 --train 0 --reload 1 --backbone hrnet_48
```

Our checkpoints are released [here,](https://drive.google.com/drive/folders/1O_i3OUTnqlkLWFu_3WKPU7YepWhItd59?usp=drive_link) and we assume you placed them (`HRNet_32_64_no_refine_24_3214.pth` or `HRNet_48_96_no_refine_45_3125.pth`) under `checkpoint/`. For more metrics (e.g., PCK), please follow the instructions in the [original repo](https://github.com/paTRICK-swk/P-STMO?tab=readme-ov-file#mpi-inf-3dhp).

