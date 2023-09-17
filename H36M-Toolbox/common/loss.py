# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def mpjpe(predicted, target, weights=None, gamma=0, return_weights=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    norm = torch.norm(predicted - target, dim=len(target.shape)-1)#.mean(axis=2).squeeze(-1)
    if weights is not None:
        norm = (weights ** gamma) * norm
        # norm = (weights.view(-1, 1, 1) ** gamma) * norm

    if return_weights:
        return torch.mean(norm), norm
    else:
        return torch.mean(norm) #, norm
    # return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def pck(pred, gt):
    error = np.linalg.norm(pred - gt, ord=2, axis=-1)
    pck = (error < 0.15).astype(np.float32).mean() * 100
    return pck  

def auc(pred, gt):
    error = np.linalg.norm(pred - gt, ord=2, axis=-1)

    thresholds = np.linspace(0., 0.15, 31)
    pck_values = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        pck_values[i] = (error < thresholds[i]).astype(np.float32).mean()

    auc = pck_values.mean() * 100
    return auc   

class kl_loss(nn.Module):
    def __init__(self, num_bins):
        super(kl_loss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=-1) #[B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='mean')
        self.num_bins = num_bins
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = self.criterion_(scores, labels)
        return loss

    def forward(self, predicted, target, weights=None, gamma=0):
        output_x, output_y, output_z = predicted
        target_x = target[:,:,:,:self.num_bins[0]]
        target_y = target[:,:,:,self.num_bins[0]:self.num_bins[0]+self.num_bins[1]]
        target_z = target[:,:,:,-self.num_bins[2]:]
        num_joints = output_x.size(2)

        loss = 0
        for idx in range(num_joints):
            loss += self.criterion(output_x[:,:,idx],target_x[:,:,idx])
            loss += self.criterion(output_y[:,:,idx],target_y[:,:,idx])
            loss += self.criterion(output_z[:,:,idx],target_z[:,:,idx])
        return loss / num_joints

# def kl_loss(predicted, target, weights=None, gamma=0):
#     LogSoftmax = nn.LogSoftmax(dim=-1)
#     loss = nn.KLDivLoss(reduction="mean")
#     return loss(LogSoftmax(predicted), target)

def mse(predicted, target, weights=None, gamma=0):
    loss = nn.MSELoss()
    return loss(predicted, target)

def cross_entropy(predicted, target, weights=None, gamma=0, return_weights=False):
    loss = nn.CrossEntropyLoss()
    return loss(predicted.permute(0, 4, 1, 2, 3), target)
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)#[0]

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))