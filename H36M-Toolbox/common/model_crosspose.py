## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import project_to_2d

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
torch.autograd.set_detect_anomaly(True)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        x = torch.cat((self.proj_q(q), self.proj_k(k), self.proj_v(v)), dim=2)
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cross_attention=False, mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attention = cross_attention
        self.mlp = mlp
        if cross_attention:
            self.norm3 = norm_layer(dim)
            self.norm4 = norm_layer(dim)
            self.attn = CrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, q=None, k=None, v=None):
        if self.cross_attention:
            x = x + self.drop_path(self.attn(self.norm1(q), self.norm3(k), self.norm4(v)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PoseTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, embed_3d=False):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        # self.embed_3d = embed_3d
        # if embed_3d:
        #     xs = torch.linspace(-1, 1, steps=5)
        #     uvd = torch.cat([x.unsqueeze(-1) for x in torch.meshgrid(xs, xs, xs)], dim=-1)
        #     # uvd *= abs(xs).view(1, 1, -1, 1)
        #     uvd[:, :, :, :2] *= abs(xs).view(1, 1, -1, 1)
        #     uvd = uvd.view(-1, 3)
        #     self.Spatial_pos_embed_3d = nn.Parameter(uvd.transpose(1, 0))
        #     # self.camera_intrinsics = nn.Parameter(torch.ones(1, 9))
        #     self.proj_3d = nn.Sequential(
        #         # nn.Linear(375, int(embed_dim_ratio*mlp_ratio)),
        #         nn.Linear(375, embed_dim),
        #         nn.ReLU(),
        #         # nn.Linear(int(embed_dim_ratio*mlp_ratio), embed_dim_ratio)
        #         nn.Linear(embed_dim , embed_dim),
        #     )
        #     self.activation = nn.Sequential(
        #         nn.Conv1d(in_channels=num_frame, out_channels=num_frame, kernel_size=1),
        #         nn.ReLU(),
        #         nn.Conv1d(in_channels=num_frame, out_channels=num_frame, kernel_size=1),
        #         nn.Sigmoid(),
        #     )

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth+1)]  # stochastic depth decay rule

        self.pose_query = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        nn.init.normal_(self.pose_query.data, std=1e-3)

        self.proj = nn.ModuleList([
            nn.Linear(embed_dim_ratio, 3),
            nn.Linear(2, embed_dim_ratio)])

        self.SA = nn.Sequential(
            nn.LayerNorm(embed_dim*2 + num_joints*2),
            nn.Linear(embed_dim*2 + num_joints*2, embed_dim)
        )

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth-1)])

        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, cross_attention=True, mlp=False)
            for i in range(int(depth+1))])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, cross_attention=True, mlp=True)
            for i in range(int(depth+1))])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed # [768, 17, 32]

        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f) # [256, 3, 544]
        return x

    def coordinate_proj(self, x, pose_query, cam):
        x = self.proj[0](rearrange(x, 'b f (w c) -> b f w c', w=17))
        x = torch.sigmoid(x).clone()
        x[:, :, :, :2] *= 2
        x[:, :, :, 2] = x[:, :, :, 2] * 5 + 3
        reference_joints_2d = project_to_2d(x, cam[:, :9])
        pose_query = self.proj[1](reference_joints_2d)
        pose_query = rearrange(pose_query, 'b f w c -> b f (w c)')
        return pose_query, reference_joints_2d

    def semantic_align(self, pose_query, reference_joints_2d, features, inputs_2d):
        b, f, _, _ = inputs_2d.shape
        return self.SA(torch.cat((pose_query, features, (reference_joints_2d-inputs_2d).view(b, f, -1)), dim=-1))

    def forward_features(self, features, inputs_2d, cam):
        b, f, _ = features.shape
        pose_query = self.pose_query.repeat(b, 1, 1)
        x = pose_query.clone()
        pose_query = rearrange(pose_query, 'b f (w c) -> b f w c', w=17)
        # x += self.Temporal_pos_embed
        # x = self.pos_drop(x)
        for blk_1, blk_2 in zip(self.Temporal_blocks, self.blocks):
            pose_query, reference_joints_2d = self.coordinate_proj(x, pose_query, cam)
            x = blk_1(x, x+pose_query, x+pose_query, x)
            pose_query = self.semantic_align(pose_query, reference_joints_2d, features, inputs_2d)
            x = blk_2(x, x+pose_query, self.Temporal_pos_embed+features, features)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def forward(self, x, cam=None): # [256, f, 17, 2]
        inputs_2d = x.clone()
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)

        # if self.embed_3d:
        #     cam = cam[:, 9:].view(-1, 3, 3)
        #     weights = self.activation(x.clone())
        #     x += weights * self.proj_3d((cam @ self.Spatial_pos_embed_3d).view(b, -1)).view(b, 1, -1)

        x = self.forward_features(x, inputs_2d, cam)
        x = self.head(x) # [256, 1, 51]

        x = x.view(b, 1, p, -1) # [256, 1, 17, 3]
        return x

