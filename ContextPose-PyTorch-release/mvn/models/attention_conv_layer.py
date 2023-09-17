import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import pickle
import time

# code for contextpose
class AttentionConv3D(nn.Module):
    pairwise_atten = {}
    def __init__(self, in_channels, out_channels, config):
        r"""
        Remarks:
            Input: (b, c_in, d_in, h_in, w_in) feature
            Output size: (b, c_out, d, h, w) feature
        """
        super(AttentionConv3D, self).__init__()
        self.num_joints = 17
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert in_channels % self.num_joints == 0 and out_channels % self.num_joints == 0, 'in_channels and out_channels must be divisible by the number of joints (17).'

        # Time for Space!
        self.convlist = nn.ModuleList([nn.Conv3d(out_channels, out_channels // self.num_joints, kernel_size=1, stride=1, padding=0, bias=False)\
            for i in range(self.num_joints)])
        for conv in self.convlist:
            nn.init.xavier_normal_(conv.weight)

        self.W_pi = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, groups=self.num_joints)
        self.atten_conv = nn.Conv3d(in_channels, self.num_joints, kernel_size=1, stride=1, padding=0, bias=False, groups=self.num_joints)

        self.connectivity = [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)]
        self.group_index = [[] for i in range(self.num_joints)]
        for i, (joint0, joint1) in enumerate(self.connectivity):
            self.group_index[joint0].append((joint1, i, 1))
            self.group_index[joint1].append((joint0, i, 0))
        
        mean_and_std_file = h5py.File(config.train.limb_length_path, 'r')
        self.mean_and_std = {'mean': np.array(mean_and_std_file['mean']), 'std': np.array(mean_and_std_file['std'])}

        self.temperature = config.model.volume_net.temperature
        self.cuboid_size = config.model.volume_net.cuboid_size

    def forward(self, x, args):
        x_pi = self.W_pi(x)
        x_flatten = x_pi.view(*x_pi.shape[:2], -1)  # (b, c_out, d*h*w)
        # compute global attention
        atten_global = self.atten_conv(x)
        atten_global_flatten = atten_global.view(*atten_global.shape[:2], -1) # (b, j, d*h*w)
        atten_global_flatten = F.softmax(atten_global_flatten, dim = 2) # (b, j, d*h*w)

        channel_per_joint = x_flatten.shape[1] // self.num_joints
        atten_global_x_group = [(x_flatten[:, joint * channel_per_joint:(joint + 1) * channel_per_joint] *\
           atten_global_flatten[:, joint:joint+1]) for joint in range(self.num_joints)] # (j, b, c_out/j, d*h*w)

        if not x.shape[2:] in AttentionConv3D.pairwise_atten.keys():
            self.calc_pairwise_atten(x.shape[2:])
        atten_pairwise = AttentionConv3D.pairwise_atten[x.shape[2:]]#(j-1, 1, d*h*w, d*h*w)

        atten_feature_group = []
        for i, (joints, mask) in enumerate(zip(self.connectivity, atten_pairwise)):
            atten_feature_group.append([])
            for joint in joints: #(joint0, joint1)
                # pairwise attention * global attention  # mask (1, d*h*w, d*h*w)
                atten_feature = torch.sum(mask.unsqueeze(0) * atten_global_x_group[joint].unsqueeze(2), dim = 3)#(b, c_out/j, d*h*w) 
                atten_feature /= (torch.sum(mask * atten_global_flatten[:, joint:joint+1], dim = 2).unsqueeze(1))   # normalization
                atten_feature = atten_feature.view(*atten_feature.shape[:2], *x.shape[2:]) #(b, c_out/j, d, h, w)           
                atten_feature_group[-1].append(atten_feature)

        atten_feature_share = torch.cat([atten_x.sum(dim=2) for atten_x in atten_global_x_group], dim = 1) #(b, c_out)

        out = []
        for joint0, conv in enumerate(self.convlist):
            atten_feature = atten_feature_share.view(*atten_feature_share.shape, 1, 1, 1).repeat(1, 1, *x.shape[2:]) 
            # (b, c_out, d, h, w)
        
            for (joint1, group, i) in self.group_index[joint0]:
                atten_feature[:, joint1 * channel_per_joint: (joint1 + 1) * channel_per_joint] = atten_feature_group[group][i]

            out.append(conv(atten_feature)) #(b, c_out / g, d, h, w)

        out = torch.cat(out, dim = 1) + x_pi

        return out, atten_global_flatten

    def calc_pairwise_atten(self, shape):
        limb_mean = torch.from_numpy(self.mean_and_std['mean'][:len(self.connectivity)]).cuda().float()#(16,)
        limb_std = torch.from_numpy(self.mean_and_std['std'][:len(self.connectivity)]).cuda().float()#(16,)
        
        xx, yy, zz = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]))
        #(d, h, w)

        grid_length = torch.tensor([self.cuboid_size / (w - 1) for w in shape])
        grid_coordinate = torch.cat((xx.unsqueeze(0) * grid_length[0],\
            yy.unsqueeze(0) * grid_length[1], zz.unsqueeze(0) * grid_length[2]), dim = 0).cuda().float()
        #(3, d, h, w)

        grid_coordinate = grid_coordinate.view(3, -1).permute(1, 0) #(d*h*w, 3)
        grid_delta = grid_coordinate.unsqueeze(1) - grid_coordinate.unsqueeze(0)
        #(d*h*w, d*h*w, 3)
    
        grid_dis = torch.norm(grid_delta, dim = 2) #(d*h*w, d*h*w)
        
        AttentionConv3D.pairwise_atten[shape] = [F.softmax(-torch.pow(grid_dis - mean, 2) / (self.temperature * 2 * torch.pow(std, 2) + 1e-1), dim = 1)\
            .unsqueeze(0) for mean, std in zip(limb_mean, limb_std)]