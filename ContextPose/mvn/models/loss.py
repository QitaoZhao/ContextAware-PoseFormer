import numpy as np

import torch
from torch import nn


def UNCERTAINTY(sigma_list, keypoints_pred, keypoints_gt):
	loss = 0.0
	diff = keypoints_pred - keypoints_gt 
	for sigma in sigma_list:
		loss_term = torch.mean(torch.norm(diff / (sigma + 1e-6), dim=len(keypoints_gt.shape)-1)) + 0.01 * torch.mean(torch.log(sigma + 1e-6))
		loss += loss_term
	return loss


class MPJPE(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, keypoints_pred, keypoints_gt):
		assert keypoints_pred.shape == keypoints_gt.shape
		return torch.mean(torch.norm(keypoints_pred - keypoints_gt, dim=len(keypoints_gt.shape)-1))


class P_MPJPE(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, keypoints_pred, keypoints_gt):
		"""
		Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
		often referred to as "Protocol #2" in many papers.
		"""
		assert keypoints_pred.shape == keypoints_gt.shape
		
		muX = np.mean(keypoints_gt, axis=1, keepdims=True)
		muY = np.mean(keypoints_pred, axis=1, keepdims=True)
		
		X0 = keypoints_gt - muX
		Y0 = keypoints_pred - muY

		normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
		normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
		np.seterr(divide='ignore', invalid='ignore')
		X0 /= normX
		Y0 /= normY

		H = np.matmul(X0.transpose(0, 2, 1), Y0)
		try:
			U, s, Vt = np.linalg.svd(H)
		except np.linalg.LinAlgError:
			return 100
		V = Vt.transpose(0, 2, 1)
		R = np.matmul(V, U.transpose(0, 2, 1))
		
		# X0 /= normX
		# Y0 /= normY

		# H = np.matmul(X0.transpose(0, 2, 1), Y0)
		# U, s, Vt = np.linalg.svd(H)
		# V = Vt.transpose(0, 2, 1)
		# R = np.matmul(V, U.transpose(0, 2, 1))

		# Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
		sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
		V[:, :, -1] *= sign_detR
		s[:, -1] *= sign_detR.flatten()
		R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

		tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

		a = tr * normX / normY # Scale
		t = muX - a*np.matmul(muY, R) # Translation
		
		# Perform rigid transformation on the input
		keypoints_pred_aligned = a*np.matmul(keypoints_pred, R) + t
		
		# Return MPJPE
		return np.mean(np.linalg.norm(keypoints_pred_aligned - keypoints_gt, axis=len(keypoints_gt.shape)-1))
	

class N_MPJPE(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, keypoints_pred, keypoints_gt):
		"""
		Normalized MPJPE (scale only), adapted from:
		https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
		"""
		assert keypoints_pred.shape == keypoints_gt.shape
		
		norm_keypoints_pred = torch.mean(torch.sum(keypoints_pred**2, dim=3, keepdim=True), dim=2, keepdim=True)
		norm_keypoints_gt = torch.mean(torch.sum(keypoints_gt*keypoints_pred, dim=3, keepdim=True), dim=2, keepdim=True)
		scale = norm_keypoints_gt / norm_keypoints_pred
		return MPJPE()(scale * keypoints_pred, keypoints_gt)#[0]

class MPJVE(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, keypoints_pred, keypoints_gt):
# def mean_velocity_error(predicted, target):
		"""
		Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
		"""
		assert keypoints_pred.shape == keypoints_gt.shape
		
		velocity_predicted = np.diff(keypoints_pred, axis=0)
		velocity_target = np.diff(keypoints_gt, axis=0)
		
		return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(keypoints_gt.shape)-1))


class KeypointsMSELoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
		dimension = keypoints_pred.shape[-1]
		loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
		loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
		return loss


class KeypointsMSESmoothLoss(nn.Module):
	def __init__(self, threshold=400):
		super().__init__()

		self.threshold = threshold

	def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
		dimension = keypoints_pred.shape[-1]
		diff = (keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity
		diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
		loss = torch.sum(diff) / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
		return loss


class KeypointsMAELoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
		dimension = keypoints_pred.shape[-1]
		loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred) * keypoints_binary_validity)
		loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
		return loss


class KeypointsL2Loss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
		loss = torch.sum(torch.sqrt(torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity, dim=2)))
		loss = loss / max(1, torch.sum(keypoints_binary_validity).item())
		return loss


class VolumetricCELoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, coord_volumes_batch, volumes_batch_pred, keypoints_gt, keypoints_binary_validity):
		loss = 0.0
		n_losses = 0

		batch_size = volumes_batch_pred.shape[0]
		for batch_i in range(batch_size):
			coord_volume = coord_volumes_batch[batch_i]
			keypoints_gt_i = keypoints_gt[batch_i]

			coord_volume_unsq = coord_volume.unsqueeze(0)
			keypoints_gt_i_unsq = keypoints_gt_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)

			dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
			dists = dists.view(dists.shape[0], -1)

			min_indexes = torch.argmin(dists, dim=-1).detach().cpu().numpy()
			min_indexes = np.stack(np.unravel_index(min_indexes, volumes_batch_pred.shape[-3:]), axis=1)

			for joint_i, index in enumerate(min_indexes):
				validity = keypoints_binary_validity[batch_i, joint_i]
				loss += validity[0] * (-torch.log(volumes_batch_pred[batch_i, joint_i, index[0], index[1], index[2]] + 1e-6))
				n_losses += 1


		return loss / n_losses


class LimbLengthError(nn.Module):
	""" Limb Length Loss: to let the """
	def __init__(self):
		super(LimbLengthError, self).__init__()
		self.CONNECTIVITY_DICT = [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)]

	def forward(self, keypoints_3d_pred, keypoints_3d_gt):
		# (b, 17, 3)

		error = 0
		for (joint0, joint1) in self.CONNECTIVITY_DICT:
			limb_pred = keypoints_3d_pred[:, joint0] - keypoints_3d_pred[:, joint1]
			limb_gt = keypoints_3d_gt[:, joint0] - keypoints_3d_gt[:, joint1]
			if isinstance(limb_pred, np.ndarray):
				limb_pred = torch.from_numpy(limb_pred)
				limb_gt = torch.from_numpy(limb_gt)
			limb_length_pred = torch.norm(limb_pred, dim = 1)
			limb_length_gt = torch.norm(limb_gt, dim = 1)
			error += torch.abs(limb_length_pred - limb_length_gt).mean().cpu()

		return float(error)/len(self.CONNECTIVITY_DICT)


