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
from mvn.models.pose_dformer_video import PoseTransformer


class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints

        # self.backbone = pose_hrnet.get_pose_net(config.model.backbone)
        # self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        # self.backbone = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=False)

        # if config.model.backbone.fix_weights:
        #     print("model backbone weights are fixed")
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False

        self.volume_net = PoseTransformer(config.model.poseformer)


    def forward(self, keypoints_2d_cpn, feats_4x, feats_8x, feats_16x, feats_32x):
    # def forward(self, images, keypoints_2d_cpn, keypoints_2d_cpn_crop):
        # [512, 3, 256, 192, 3] [512, 3, 17, 2] [512, 3, 17, 2]
        # b, f, h, w, c = images.shape
        # images = images.view(-1, h, w, c)
        keypoints_2d_cpn = keypoints_2d_cpn.reshape(-1, 17, 2)
        # keypoints_2d_cpn_crop = keypoints_2d_cpn_crop.reshape(-1, 17, 2)
        feats_4x = feats_4x.reshape(-1, 17, 32)
        feats_8x = feats_8x.reshape(-1, 17, 32)
        feats_16x = feats_16x.reshape(-1, 17, 32)
        feats_32x = feats_32x.reshape(-1, 17, 32)

        device = keypoints_2d_cpn.device
        # images = images.permute(0, 3, 1, 2).contiguous()

        # keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        # keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        # forward backbone
        # features_list = self.backbone(images) 
        # features_list.pop(3)
        # torch.Size([4, 32, 64, 48])
        # torch.Size([4, 64, 32, 24])
        # torch.Size([4, 128, 16, 12])
        # torch.Size([4, 256, 8, 6])

        # keypoints_3d = self.volume_net(keypoints_2d_cpn, features_sampled_list)
        keypoints_3d = self.volume_net(keypoints_2d_cpn, [feats_4x, feats_8x, feats_16x, feats_32x])

        return keypoints_3d

