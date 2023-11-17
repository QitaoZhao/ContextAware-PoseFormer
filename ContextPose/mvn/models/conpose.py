import torch
from torch import nn

from mvn.models import pose_hrnet
from mvn.models.pose_dformer import PoseTransformer


class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
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

        self.volume_net = PoseTransformer(config.model.poseformer)


    def forward(self, images, keypoints_2d_cpn, keypoints_2d_cpn_crop):
        device = keypoints_2d_cpn.device
        images = images.permute(0, 3, 1, 2).contiguous()

        keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        # forward backbone
        features_list = self.backbone(images) 

        keypoints_3d = self.volume_net(keypoints_2d_cpn, keypoints_2d_cpn_crop, features_list)

        return keypoints_3d

