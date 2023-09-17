import _init_path
from mvn import datasets
from mvn.utils.cfg import config, update_config
import numpy as np

from easydict import EasyDict as edict
import yaml
import os
import sys
import time
from tqdm import tqdm
import h5py

actual_joints = {
        0: "right_ankle",
        1: "right_knee",
        2: "right_hip",
        3: "left_hip",
        4: "left_knee",
        5: "left_ankle",
        6: "root",
        7: "belly",
        8: "neck",
        9: "head",
        10: "right_wrist",
        11: "right_elbow",
        12: "right_shoulder",
        13: "left_shoulder",
        14: "left_elbow",
        15: "left_wrist",
        16: "nose"
        }

CONNECTIVITY_DICT = {
    'cmu': [(0, 2), (0, 9), (1, 0), (1, 17), (2, 12), (3, 0), (4, 3), (5, 4), (6, 2), (7, 6), (8, 7), (9, 10), (10, 11), (12, 13), (13, 14), (15, 1), (16, 15), (17, 18)],
    'coco': [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (5, 11), (6, 12), (11, 12)],
    "mpii": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (8, 12), (8, 13), (10, 11), (11, 12), (13, 14), (14, 15)],
    "human36m": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)],
    "kth": [(0, 1), (1, 2), (5, 4), (4, 3), (6, 7), (7, 8), (11, 10), (10, 9), (2, 3), (3, 9), (2, 8), (9, 12), (8, 12), (12, 13)],
}

dataset_type = 'h36m'
if dataset_type == 'h36m':
    h36m_root_path = sys.argv[1]

    ###################################### TRAIN ######################################
    update_config('experiments/human36m/train/human36m_vol_softmax_single.yaml')

    train_dataset = eval('datasets.keypoint_human36m')(
        root=config.dataset.root,
        pred_results_path=config.train.pred_results_path,
        train=True,
        test=False,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.train_labels_path,
        with_damaged_actions=config.train.with_damaged_actions,
        scale_bbox=config.train.scale_bbox,
        kind=config.kind,
        undistort_images=config.train.undistort_images,
        ignore_cameras=config.train.ignore_cameras,
        crop=config.train.crop
    )

    print('=> computing limb length WITH limb connections in train_dataset ...')
    limb_length = np.zeros((len(train_dataset), len(CONNECTIVITY_DICT["human36m"])))
    for index, train_data in enumerate(tqdm(train_dataset)):
        keypoints_3d = train_data['keypoints_3d']
        for j, (joint1, joint2) in enumerate(CONNECTIVITY_DICT["human36m"]):
            limb_length[index, j] = np.linalg.norm(keypoints_3d[joint1, :3] - keypoints_3d[joint2, :3])

    train_mean = limb_length.mean(axis = 0)
    train_std = limb_length.std(axis = 0)
    for i, (joint1, joint2) in enumerate(CONNECTIVITY_DICT["human36m"]):
        # print(actual_joints[joint1], actual_joints[joint2], train_mean[i], train_std[i]) 
        print(joint1, joint2, train_mean[i], train_std[i]) 

    print('=> writing TRAIN limb length to mean_and_std_limb_length.h5...')
    limb_length_file = h5py.File(os.path.join(h36m_root_path, "extra/mean_and_std_limb_length.h5"), 'w')
    limb_length_file['mean'] = train_mean
    limb_length_file['std'] = train_std
    limb_length_file.close()
