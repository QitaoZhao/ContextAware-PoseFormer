from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn
import torch.nn.functional as F

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models import pose_resnet
from mvn.models.v2v_net import V2VNet



class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_net.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_net.volume_softmax
        self.volume_multiplier = config.model.volume_net.volume_multiplier
        self.volume_size = config.model.volume_net.volume_size

        self.cuboid_size = config.model.volume_net.cuboid_size

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.volume_net.use_gt_pelvis

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.dataset.transfer_cmu_to_human36m

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        self.use_feature = config.model.volume_net.use_feature_v2v


        if config.model.backbone.fix_weights:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif self.use_feature:
            for p in self.backbone.final_layer.parameters():
                p.requires_grad = False

        v2v_in_channel = self.num_joints
        if self.use_feature:
            self.process_features = nn.Sequential(
                nn.Conv2d(256, 32, 1)
            )
            v2v_in_channel = 32

        self.volume_net = V2VNet(v2v_in_channel, self.num_joints, config)

    def build_coord_volume(self, coord_volume_size, position, sizes, base_point, theta, axis, device):
        # build coord volume
        xxx, yyy, zzz = torch.meshgrid(torch.arange(coord_volume_size, device=device), torch.arange(coord_volume_size, device=device), torch.arange(coord_volume_size, device=device))
        grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
        grid = grid.reshape((-1, 3))

        grid_coord = torch.zeros_like(grid)
        grid_coord[:, 0] = position[0] + (sizes[0] / (coord_volume_size - 1)) * grid[:, 0]
        grid_coord[:, 1] = position[1] + (sizes[1] / (coord_volume_size - 1)) * grid[:, 1]
        grid_coord[:, 2] = position[2] + (sizes[2] / (coord_volume_size - 1)) * grid[:, 2]

        coord_volume = grid_coord.reshape(coord_volume_size, coord_volume_size, coord_volume_size, 3)

        center = torch.from_numpy(base_point).type(torch.float).to(device)

        # rotate
        coord_volume = coord_volume - center
        coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
        coord_volume = coord_volume + center

        # transfer
        if self.transfer_cmu_to_human36m:  # different world coordinates
            coord_volume = coord_volume.permute(0, 2, 1, 3)
            inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
            coord_volume = coord_volume.index_select(1, inv_idx)

        return coord_volume 

    def calc_ga_mask(self, keypoints_3d_gt, coord_volume, sigma=2):
        """
        :param keypoints_3d_gt:  (batch_size, n_joints, 3)
        :param coord_volume: (batch_size, d, h, w, 3)
        :return: ga_mask_gt: (batch_size, n_joints, d, h, w)
        """
        shape = coord_volume.shape
        delta = coord_volume.view(shape[0], 1, -1, 3) - keypoints_3d_gt.unsqueeze(2) # (b, j, d*h*w, 3)
        dist = torch.norm(delta, dim = 3) # (b, j, d*h*w)
        sigma *= torch.norm(coord_volume[0, 0, 0, 1] - coord_volume[0, 0, 0, 0])
        tmp_size = sigma * 2.5
        ga_mask_gt = F.softmax((-torch.square(dist) / (2 * sigma ** 2)), dim=2).float() # (b, j, d*h*w)

        return ga_mask_gt


    def forward(self, images, proj_matricies, batch, keypoints_3d_gt):
        device = images.device
        batch_size, n_views = images.shape[:2]   # images [batch_size, n_views, 3, 384, 384]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])   # images [batch_size*n_views, 3, 384, 384]

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])    # images [batch_size, n_views, 3, 384, 384]
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])    # heatmaps [batch_size, n_views, 17, 96, 96]
        features = features.view(batch_size, n_views, *features.shape[1:])    # features [batch_size, n_views, 256, 96, 96]

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]    # 17

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        coord_volumes_aux = torch.zeros(batch_size, self.volume_size//4, self.volume_size//4, self.volume_size//4, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sizes = np.array([self.cuboid_size, self.cuboid_size, self.cuboid_size])
            aux_sizes = sizes - 3 * sizes / (self.volume_size - 1)
            position = base_point - sizes / 2
            cuboid = volumetric.Cuboid3D(position, sizes)

            cuboids.append(cuboid)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            # build coord volume
            coord_volumes[batch_i] = self.build_coord_volume(self.volume_size, position, sizes, base_point, theta, axis, device)  
            coord_volumes_aux[batch_i] = self.build_coord_volume(self.volume_size//4, position, aux_sizes, base_point, theta, axis, device)    

        # compute gt global attention, using keypoints_3d_gt
        ga_mask_gt = self.calc_ga_mask(keypoints_3d_gt, coord_volumes_aux)

        # process features before unprojecting
        if self.use_feature:
            features = features.view(-1, *features.shape[2:])    # features [batch_size*n_views, 256, 96, 96]
            features = self.process_features(features)      # conv2d 1x1 kernel [256 -> 32]
            features = features.view(batch_size, n_views, *features.shape[1:])    # features [batch_size, n_views, 32, 96, 96]

            v2v_input = features
        else:
            v2v_input = heatmaps

        # lift to volume
        volumes = op.unproject_heatmaps(v2v_input, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)    # volumes [batch_size, 32, 64, 64, 64]


        # integral 3d
        volumes, atten_global = self.volume_net(volumes, None)      # volumes [batch_size, 17, 64, 64, 64]
        voxel_keypoints_3d, _ = op.integrate_tensor_3d(volumes * self.volume_multiplier, softmax=self.volume_softmax)
        # voxel_3d: keypoints_3d in volumes [batch_size, 17, 3]
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)       # vol_keypoints_3d [batch_size, 17, 3]


        return voxel_keypoints_3d, vol_keypoints_3d, heatmaps, volumes, ga_mask_gt, atten_global, vol_confidences, cuboids, coord_volumes, base_points

