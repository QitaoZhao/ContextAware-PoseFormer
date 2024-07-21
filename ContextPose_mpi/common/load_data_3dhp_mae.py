
import torch.utils.data as data
import numpy as np

from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generator_3dhp import ChunkedGenerator

class Fusion(data.Dataset):
    def __init__(self, opt, root_path, train=True, MAE=False):
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        self.MAE=MAE
        if self.train:
            self.poses_train, self.poses_train_2d, self.poses_train_2d_crop = self.prepare_data(opt.root_path, train=True)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, None, self.poses_train,
                                              self.poses_train_2d, self.poses_train_2d_crop, None, chunk_length=self.stride, pad=self.pad,
                                              augment=False, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all, MAE=MAE, train = True)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.poses_test, self.poses_test_2d, self.poses_test_2d_crop, self.valid_frame = self.prepare_data(opt.root_path, train=False)
            # self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
            #                                                                     subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, None, self.poses_test,
                                              self.poses_test_2d, self.poses_test_2d_crop, self.valid_frame,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, MAE=MAE, train = False)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, path, train=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_poses_2d_crop = {}
        valid_frame = {}

        self.kps_left, self.kps_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
        self.joints_left, self.joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]

        if train == True:
            data = np.load("dataset/data_train_3dhp.npz",allow_pickle=True)['data'].item()
            for seq in data.keys():
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    subject_name, seq_name = seq.split(" ")

                    data_3d = anim['data_3d']
                    data_3d[:, :14] -= data_3d[:, 14:15]
                    data_3d[:, 15:] -= data_3d[:, 14:15]
                    out_poses_3d[(subject_name, seq_name, cam)] = data_3d

                    data_2d = anim['data_2d']
                    data_2d_crop = anim['data_2d_crop']

                    data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
                    out_poses_2d[(subject_name, seq_name, cam)] = data_2d
                    out_poses_2d_crop[(subject_name, seq_name, cam)] = data_2d_crop

            return out_poses_3d, out_poses_2d, out_poses_2d_crop
        else:
            data = np.load("dataset/data_test_3dhp.npz", allow_pickle=True)['data'].item()
            for seq in data.keys():

                anim = data[seq]

                valid_frame[seq] = anim["valid"]

                data_3d = anim['data_3d']
                data_3d[:, :14] -= data_3d[:, 14:15]
                data_3d[:, 15:] -= data_3d[:, 14:15]
                out_poses_3d[seq] = data_3d

                data_2d = anim['data_2d']
                data_2d_crop = anim['data_2d_crop']

                if seq == "TS5" or seq == "TS6":
                    width = 1920
                    height = 1080
                else:
                    width = 2048
                    height = 2048
                data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
                out_poses_2d[seq] = data_2d
                out_poses_2d_crop[seq] = data_2d_crop

            return out_poses_3d, out_poses_2d, out_poses_2d_crop, valid_frame

    def __len__(self):
        return len(self.generator.pairs)
        #return 200

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
        # ['S7' 'Seq2' '6'] 10782 10783 False False

        if self.MAE:
            pass
            # cam, input_2D, seq, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip,
            #                                                                           reverse)
            # if self.train == False and self.test_aug:
            #     _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
            #     input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
        else:
            cam, gt_3D, input_2D, input_2D_crop, img, seq, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)

            # if self.train == False and self.test_aug:
            #     _, _, input_2D_aug, _, _, _, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
            #     _, _, _, input_2D_crop_aug, _, _, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
            #     _, _, _, _, img_aug, _, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
            #     input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            #     input_2D_crop = np.concatenate((np.expand_dims(input_2D_crop,axis=0),np.expand_dims(input_2D_crop_aug,axis=0)),0)
            #     img = np.concatenate((np.expand_dims(img,axis=0),np.expand_dims(img,axis=0)),0)
            
        bb_box = np.array([0, 0, 1, 1])

        scale = float(1.0)

        if self.MAE:
            if self.train == True:
                return cam, input_2D_update, seq, subject, scale, bb_box, cam_ind
            else:
                return cam, input_2D_update, seq, scale, bb_box
        else:
            if self.train == True:
                return cam, gt_3D, input_2D, input_2D_crop, img, seq, subject, scale, bb_box, cam_ind
            else:
                return cam, gt_3D, input_2D, input_2D_crop, img, seq, scale, bb_box



