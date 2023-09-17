from copy import deepcopy
import numpy as np
import pickle
import random
from time import time

from scipy.optimize import least_squares

import torch
from torch import nn
import torch.nn.functional as F

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models import pose_hrnet, pose_resnet
from mvn.models.networks import network
from mvn.models.cpn.test_config import cfg
from mvn.models.v2v_net import V2VNet
from mvn.models.pose_dformer import PoseTransformer


class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints

        # self.backbone = pose_hrnet.get_pose_net(config.model.backbone)
        # self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        # output_shape = (96, 72) for 384 * 288 CPN model
        # self.backbone = network.__dict__[cfg.model]((96,72), cfg.num_class, pretrained=False)
        self.backbone = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=False)

        if config.model.backbone.fix_weights:
            print("model backbone weights are fixed")
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.volume_net = PoseTransformer(config.model.poseformer)


    def forward(self, images, keypoints_2d_cpn, keypoints_2d_cpn_crop):
        # start = time()
        device = keypoints_2d_cpn.device
        images = images.permute(0, 3, 1, 2).contiguous()

        keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        # forward backbone
        features_list = self.backbone(images) 
        # features_list.pop(3)
        # torch.Size([4, 32, 64, 48])
        # torch.Size([4, 64, 32, 24])
        # torch.Size([4, 128, 16, 12])
        # torch.Size([4, 256, 8, 6])

        # features_sampled_list = [
        #     F.grid_sample(features, keypoints_2d_cpn_crop.unsqueeze(-2), align_corners=True).squeeze().permute(0, 2, 1).contiguous() \
        #     for features in features_list]
        # torch.Size([4, 17, 32])
        # torch.Size([4, 17, 64])
        # torch.Size([4, 17, 128])
        # torch.Size([4, 17, 256])

        # keypoints_3d = self.volume_net(keypoints_2d_cpn, features_sampled_list)
        keypoints_3d = self.volume_net(keypoints_2d_cpn, keypoints_2d_cpn_crop, features_list)

        # print("HRNet:", ckpt_2-ckpt_1)
        # print("PoseFormer:", end-ckpt_2)
        # print("total:", end-start)
        return keypoints_3d

