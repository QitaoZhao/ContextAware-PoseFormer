from copy import deepcopy
import numpy as np
import pickle
import random
from time import time

import torch
from torch import nn
import torch.nn.functional as F

from model import pose_hrnet
from model.pose_dformer import PoseTransformer


class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints

        self.backbone = pose_hrnet.get_pose_net(config.model.backbone)

        if config.model.backbone.fix_weights:
            print("model backbone weights are fixed")
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.volume_net = PoseTransformer(config.model.poseformer)


    def forward(self, images, keypoints_2d_cpn, keypoints_2d_cpn_crop):
        device = images.device
        images = images.permute(0, 3, 1, 2).contiguous()

        keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        # forward backbone
        features_list = self.backbone(images) 
        keypoints_3d = self.volume_net(keypoints_2d_cpn, keypoints_2d_cpn_crop, features_list)

        return keypoints_3d

