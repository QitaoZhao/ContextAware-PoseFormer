import pickle
import h5py
import cv2
import numpy as np
import os.path as osp
from scipy.io import loadmat
from subprocess import call
import os
from os import makedirs
# from spacepy import pycdf
from tqdm import tqdm
import cdflib
import sys
from common.camera import *

from transform import get_affine_transform, affine_transform, \
    normalize_screen_coordinates, _infer_box, _weak_project 
from metadata import load_h36m_metadata
metadata = load_h36m_metadata()

def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0
    tl_joint[1] -= 900.0
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0
    br_joint[1] += 1100.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d

if __name__ == '__main__':
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    action_list = [x for x in range(2, 17)]
    subaction_list = [x for x in range(1, 3)]
    camera_list = [x for x in range(1, 5)]

    train_list = [1, 5, 6, 7, 8]
    test_list = [9, 11]

    joint_idx = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]

    print('Loading 2D detections...')
    keypoints = np.load('data/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    keypoints = keypoints['positions_2d'].item()

    with open('camera_data.pkl', 'rb') as f:
        camera_data = pickle.load(f)

    train_db = []
    test_db = []
    cnt = 0
    for s in subject_list:
        for a in action_list:
            for sa in subaction_list:
                for c in camera_list:
                    if c != 1:
                        continue

                    camera = camera_data[(s, c)]
                    camera_dict = {}
                    camera_dict['R'] = camera[0]
                    camera_dict['T'] = camera[1]
                    camera_dict['fx'] = camera[2][0]
                    camera_dict['fy'] = camera[2][1]
                    camera_dict['cx'] = camera[3][0]
                    camera_dict['cy'] = camera[3][1]
                    camera_dict['k'] = camera[4]
                    camera_dict['p'] = camera[5]


                    subdir_format = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'
                    subdir = subdir_format.format(s, a, sa, c)

                    print(subdir)

                    basename = metadata.get_base_filename('S{:d}'.format(s), '{:d}'.format(a), '{:d}'.format(sa), metadata.camera_ids[c-1])
                    annotname = basename + '.cdf'

                    print(s, a, sa)
                    print(basename)

                    subject = 'S' + str(s)
                    # annofile3d = osp.join('extracted', subject, 'Poses_D3_Positions_mono_universal', annotname)
                    annofile3d_camera = osp.join('extracted', subject, 'Poses_D3_Positions_mono', annotname)
                    annofile2d = osp.join('extracted', subject, 'Poses_D2_Positions', annotname)

                    # with pycdf.CDF(annofile3d) as data:
                    # data = cdflib.CDF(annofile3d)
                    # pose3d = np.array(data.varget("Pose"))
                    # pose3d = np.reshape(pose3d, (-1, 32, 3))

                    # with pycdf.CDF(annofile3d_camera) as data:
                    data = cdflib.CDF(annofile3d_camera)
                    pose3d_camera = np.array(data.varget("Pose"))
                    pose3d_camera = np.reshape(pose3d_camera, (-1, 32, 3))#[:, joint_idx] / 1000.0

                    if basename.split('.')[0] not in keypoints['S{:d}'.format(s)].keys():
                        if "TakingPhoto" in basename:
                            basename = basename.replace("TakingPhoto", "Photo")
                        elif "WalkingDog" in basename:
                            basename = basename.replace("WalkingDog", "WalkDog")
                        else:
                            print(basename.split('.')[0] + " is missing!")
                            continue

                    # with pycdf.CDF(annofile2d) as data:
                    data = cdflib.CDF(annofile2d)
                    pose2d_gt = np.array(data.varget("Pose"))
                    pose2d_gt = np.reshape(pose2d_gt, (-1, 32, 2))
                    pose2d_cpn = keypoints['S{:d}'.format(s)][basename.split('.')[0]][c-1]

                    nposes = min(pose3d_camera.shape[0], pose2d_gt.shape[0])
                    if pose2d_cpn.shape[0] > nposes:
                        pose2d_cpn = pose2d_cpn[:nposes]
                    assert pose2d_gt.shape[0] == pose2d_cpn.shape[0] 
                    assert pose2d_cpn.shape[0] == pose3d_camera.shape[0]
                    image_format = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}_{:06d}.jpg'

                    for i in tqdm(range(nposes)):
                        datum = {}
                        imagename = image_format.format(s, a, sa, c, i+1)
                        imagepath = osp.join(subdir, imagename)
                        if not osp.isfile(osp.join('images', imagepath)):
                            print(osp.join('images', imagepath))
                            print(nposes)
                        if osp.isfile(osp.join('images', imagepath)):

                            datum['image'] = imagepath
                            datum['joints_2d_gt'] = pose2d_gt[i, joint_idx, :]
                            datum['joints_2d_cpn'] = pose2d_cpn[i]
                            # datum['joints_3d'] = pose3d[i, joint_idx, :]
                            datum['joints_3d'] = pose3d_camera[i, joint_idx, :]
                            datum['joints_vis'] = np.ones((17, 3))
                            datum['video_id'] = cnt
                            datum['image_id'] = i+1
                            datum['subject'] = s
                            datum['action'] = a
                            datum['subaction'] = sa
                            datum['camera_id'] = c-1
                            datum['source'] = 'h36m'
                            datum['camera'] = camera_dict
                            datum['nposes'] = nposes

                            box = _infer_box(datum['joints_3d'], camera_dict, 0)
                            center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                            scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
                            datum['center'] = center
                            datum['scale'] = scale
                            datum['box'] = box

                            data_numpy = cv2.imread(
                                osp.join('images', imagepath), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                                )
                            h, w, _ = data_numpy.shape
                            pose2d_gt_crop = np.zeros([17, 2], dtype='float32')
                            pose2d_cpn_crop = np.zeros([17, 2], dtype='float32')

                            trans = get_affine_transform(center, scale, 0, [192, 256])

                            for j in range(17):
                                pose2d_gt_crop[j] = affine_transform(datum['joints_2d_gt'][j], trans)
                                pose2d_cpn_crop[j] = affine_transform(datum['joints_2d_cpn'][j], trans)

                            datum['joints_2d_gt_crop'] = pose2d_gt_crop
                            datum['joints_2d_cpn_crop'] = pose2d_cpn_crop

                            datum['joints_2d_gt'] = normalize_screen_coordinates(datum['joints_2d_gt'][..., :2], w=w, h=h)
                            datum['joints_2d_cpn'] = normalize_screen_coordinates(datum['joints_2d_cpn'][..., :2], w=w, h=h)
                            datum['joints_3d'] = datum['joints_3d'] / 1000.0

                            if s in train_list:
                                train_db.append(datum)
                            else:
                                test_db.append(datum)

                    cnt += 1

    with open('h36m_train.pkl', 'wb') as f:
        pickle.dump(train_db, f)

    with open('h36m_validation.pkl', 'wb') as f:
        pickle.dump(test_db, f)






