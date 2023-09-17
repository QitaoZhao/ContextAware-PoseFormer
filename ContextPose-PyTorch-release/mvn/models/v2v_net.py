# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_conv_layer import AttentionConv3D
import argparse
from torch.nn.parallel import DistributedDataParallel
import os


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)

class AttenRes3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, config):
        super(AttenRes3DBlock, self).__init__()
        self.atten_conv = AttentionConv3D(in_planes, out_planes, config)
        self.res_branch = nn.Sequential(
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x, args):
        res, atten_global = self.atten_conv(x, args)
        res = self.res_branch(res)
        skip = self.skip_con(x)
        return F.relu(res + skip, True), atten_global


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )


    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class EncoderDecorder(nn.Module):
    def __init__(self, config):
        super(EncoderDecorder, self).__init__()
        self.att_channels = config.model.volume_net.att_channels

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(34, 51)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(51, self.att_channels)

        self.mid_res = AttenRes3DBlock(self.att_channels, self.att_channels, config)

        self.decoder_res2 = Res3DBlock(self.att_channels, self.att_channels)
        self.decoder_upsample2 = Upsample3DBlock(self.att_channels, 51, 2, 2)
        self.decoder_res1 = Res3DBlock(51, 51)
        self.decoder_upsample1 = Upsample3DBlock(51, 34, 2, 2)

        self.skip_res1 = Res3DBlock(34, 34)
        self.skip_res2 = Res3DBlock(51, 51)

    def forward(self, x, args):
        skip_x1 = self.skip_res1(x)     
        x = self.encoder_pool1(x)       
        x = self.encoder_res1(x)        
        skip_x2 = self.skip_res2(x)     
        x = self.encoder_pool2(x)       
        x = self.encoder_res2(x)        

        x, atten_global = self.mid_res(x, args)       
        x = self.decoder_res2(x)        
        x = self.decoder_upsample2(x)   
        x = x + skip_x2
        x = self.decoder_res1(x)        
        x = self.decoder_upsample1(x)   
        x = x + skip_x1

        return x, atten_global

class V2VNet(nn.Module):
    def __init__(self, input_channels, output_channels, config):
        super(V2VNet, self).__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 17, 3),   
            Res3DBlock(17, 34),    
        )

        self.encoder_decoder = EncoderDecorder(config)   

        self.output_layer = nn.Conv3d(34, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x, args):
        x = self.front_layers(x)
        x, atten_global = self.encoder_decoder(x, args)     
        x = self.output_layer(x)        
        return x, atten_global

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
