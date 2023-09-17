## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


# torch.autograd.set_detect_anomaly(True)


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
        if len(x.shape) == 4:
            x = self.fc1(x.permute(0, 2, 3, 1))
        else:
            x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if len(x.shape) == 4:
            x = self.fc2(x).permute(0, 3, 1, 2)
        else:
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


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.f_c = nn.Linear(dim, heads * head_dim)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.proj_c = nn.Linear(heads * head_dim, out_dim)
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.v_c = nn.Linear(dim, heads * head_dim)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        # self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        # self.fold_w = fold_w
        # self.fold_h = fold_h
        # self.return_center = return_center

    def forward(self, x, ref):  # x: [b,c,w,h], c: [b, p, c]
        center_features = F.grid_sample(x, ref.unsqueeze(-2), align_corners=True).squeeze().permute(0, 2, 1).contiguous()

        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        # if self.fold_w > 1 and self.fold_h > 1:
        #     # split the big feature maps to small local regions to reduce computations.
        #     b0, c0, w0, h0 = x.shape
        #     assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
        #         f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
        #     x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
        #                   f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
        #     value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        # centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        # value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]

        centers = self.f_c(center_features)
        centers = rearrange(centers, "b p (e c) -> (b e) p c", e=self.heads)
        value_centers = self.v_c(center_features)
        value_centers = rearrange(value_centers, "b p (e c) -> (b e) p c", e=self.heads)
        # b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers,
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                    mask.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        c = rearrange(out, "(b e) m c -> b m (e c)", e=self.heads)
        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
        out = rearrange(out, "(b e) (w h) c -> b (e c) w h", e=self.heads,w=w)

        # if self.return_center:
            # out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        # else:
            # dispatch step, return to each point in a cluster
            # out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            # out = rearrange(out, "(b e) (w h) c -> b (e c) w h", e=self.heads,w=w)
            # out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)

        # if self.fold_w > 1 and self.fold_h > 1:
        #     # recover the splited regions back to big feature maps if use the region partition.
        #     out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        # out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        c = self.proj_c(c) 

        return out, c


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=False, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False):

        super().__init__()

        self.norm1 = nn.ModuleList([norm_layer(dim)] * 4)
        # self.norm12 = nn.ModuleList([nn.LayerNorm(dim)] * 4) 
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = nn.ModuleList([
            Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
            fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim, return_center=False)] * 4)
        self.norm2 = nn.ModuleList([norm_layer(dim)] * 4)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.ModuleList([
            Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)] * 4)

        # The following two techniques are useful to train deep ContextClusters.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((4, dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((4, dim)), requires_grad=True)

    def forward(self, ref, features_list):
        xs = []
        cs = []
        for idx in range(len(features_list)):
            x = features_list[idx]
            # x, c = features_list[idx], center_features[:, idx]

            if self.use_layer_scale:
                x_, c = self.drop_path(
                    self.layer_scale_1(torch.tensor([idx]).to(dtype=torch.long, device=x.device)).unsqueeze(-1).unsqueeze(-1)
                    * self.token_mixer[idx](self.norm11[idx](x), self.norm12[idx](c)))
                x = x + x_
                x = x + self.drop_path(
                    self.layer_scale_2(torch.tensor([idx]).to(dtype=torch.long, device=x.device)).unsqueeze(-1).unsqueeze(-1)
                    * self.mlp[idx](self.norm2[idx](x)))
            else:
                ret = self.token_mixer[idx](self.norm1[idx](x), ref)
                x = x + self.drop_path(ret[0])
                x = x + self.drop_path(self.mlp[idx](self.norm2[idx](x)))

            xs.append(x)
            cs.append(ret[1])

        cs = torch.stack(cs, dim=1) 
            # center_features[:, idx] = c
            # features_list[idx] = x

        return cs, xs


class PoseTransformer(nn.Module):
    def __init__(self, config, num_frame=1, num_joints=17, in_chans=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
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
        embed_dim_ratio = config.embed_dim_ratio
        depth = config.depth
        # embed_dim = 2048
        # embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3    #### output dimension is num_joints * 3
        self.levels = config.levels
        embed_dim = embed_dim_ratio * (self.levels+1)

        ### spatial patch embedding
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.patch_embed = nn.ModuleList([
            nn.Conv2d(32+2, embed_dim_ratio, kernel_size=1, stride=4),
            nn.Conv2d(64+2, embed_dim_ratio, kernel_size=1, stride=2),
            nn.Conv2d(128+2, embed_dim_ratio, kernel_size=1, stride=1),
            nn.Conv2d(256+2, embed_dim_ratio, kernel_size=1, stride=1)])

        # self.center_embed = nn.ModuleList([
        #     nn.Linear(32+2, embed_dim_ratio),
        #     nn.Linear(64+2, embed_dim_ratio),
        #     nn.Linear(128+2, embed_dim_ratio),
        #     nn.Linear(256+2, embed_dim_ratio)])

        # self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, self.levels+1, num_joints, embed_dim_ratio))
        # self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.joint_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.res_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.cluster_blocks = nn.ModuleList([ClusterBlock(dim=embed_dim_ratio) for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        # self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        # self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def get_center_features(self, ref, features_list):
        device = ref.device
        # ref: [b, p, 2]
        ref[..., :2] /= torch.tensor([192 - 1.0, 256 - 1.0], device=device)
        ref_ = ref.clone()
        ref_ = ref_ * 2 - 1

        center_features = [
            F.grid_sample(features, ref_.unsqueeze(-2), align_corners=True).squeeze().permute(0, 2, 1).contiguous() \
            for features in features_list]

        ref -= 0.5
        center_features = [
            self.center_embed[idx](torch.cat([feature, ref], dim=-1)) \
            for idx, feature in enumerate(center_features)]

        return center_features

    def forward_embeddings(self, x, idx):
        _, c, img_w, img_h = x.shape
        # print(f"det img size is {img_w} * {img_h}")
        # register positional information buffer.
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)
        x = self.patch_embed[idx](torch.cat([x, pos], dim=1))
        return x

    def forward(self, keypoints_2d, ref, features_list):
        b, p, c = keypoints_2d.shape
        device = keypoints_2d.device
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data

        ref[..., :2] /= torch.tensor([192 - 1.0, 256 - 1.0], device=device)
        ref = ref * 2 - 1

        x = self.coord_embed(keypoints_2d).unsqueeze(1)
        # center_features = self.get_center_features(ref, features_list)
        # x = torch.stack([x,*center_features], dim=1) # [b, 5, p, c]

        features_list = [self.forward_embeddings(feature, idx) \
                         for idx, feature in enumerate(features_list)]

        # x = torch.stack([x,*features_ref_list], dim=1) # [b, p, 4, c]

        # x += self.Spatial_pos_embed
        # x = self.pos_drop(x)
        for index, (J_blk, R_blk, C_blk) in enumerate(zip(self.joint_blocks, self.res_blocks, self.cluster_blocks)):
            center_features, features_list = C_blk(ref, features_list)
            x = torch.cat([x,center_features], dim=1)
            x = rearrange(x, 'b l p c -> (b p) l c')
            x = R_blk(x)
            x = rearrange(x, '(b p) l c -> b p (l c)', p=p)
            x = J_blk(x)
            x = rearrange(x, 'b p (l c) -> b l p c', l=self.levels+1)
            if index != 3:
                x, center_features = x[:, :1], x[:, 1:]

        x = self.Spatial_norm(x)
        x = rearrange(x, 'b l p c -> b p (l c)')

        x = self.head(x).view(b, 1, p, -1)
        return x
