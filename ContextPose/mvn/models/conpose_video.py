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
from mvn.models.pose_dformer_video import PoseTransformer_video


class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, frame, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints

        # HRNet
        self.backbone = pose_hrnet.get_pose_net(config.model.backbone)
        # ResNet
        # self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        # CPN
        # output_shape = (96, 72) for 384 * 288 CPN model
        # self.backbone = network.__dict__[cfg.model]((96,72), cfg.num_class, pretrained=False)
        # self.backbone = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=False)

        if config.model.backbone.fix_weights:
            print("model backbone weights are fixed")
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.volume_net = PoseTransformer_video(config.model.poseformer, num_frame=frame)


    def forward(self, images, keypoints_2d_cpn, keypoints_2d_cpn_crop):
        device = keypoints_2d_cpn.device

        # images:  torch.Size([512, 5, 256, 192, 3])
        # keypoints_2d_cpn:  torch.Size([512, 5, 17, 2])
        # keypoints_2d_cpn_crop:  torch.Size([512, 17, 2])
        B, f, h, w, c = images.shape

        images = images.permute(0, 1, 4, 2, 3).contiguous()

        images = images.view(B*f, c, h, w)


        keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        # forward backbone
        features_list = self.backbone(images) 

        keypoints_3d = self.volume_net(keypoints_2d_cpn, keypoints_2d_cpn_crop, features_list)

        return keypoints_3d

