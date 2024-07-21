import os
import glob
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.cfg import config, update_config, update_dir
from common.load_data_3dhp_mae import Fusion
from model.conpose import VolumetricTriangulationNet

from thop import clever_format
from thop.profile import profile
import scipy.io as scio

opt = opts().parse()

def train(opt, actions, train_loader, model, optimizer, epoch):
	return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
	with torch.no_grad():
		return step('test',  opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
	if split == 'train':
		model.train()
	else:
		model.eval()

	loss_all = {'loss': AccumLoss()}
	error_sum = AccumLoss()
	error_sum_test = AccumLoss()

	action_error_sum = define_error_list(actions)
	action_error_sum_post_out = define_error_list(actions)
	action_error_sum_MAE = define_error_list(actions)

	joints_left = [5, 6, 7, 11, 12, 13]
	joints_right = [2, 3, 4, 8, 9, 10]

	data_inference = {}

	for i, data in enumerate(tqdm(dataLoader, 0)):

		if False:
			pass

		else:
			if split == "train":
				batch_cam, gt_3D, input_2D, input_2D_crop, img, seq, subject, scale, bb_box, cam_ind = data
			else:
				batch_cam, gt_3D, input_2D, input_2D_crop, img, seq, scale, bb_box = data

			[input_2D, input_2D_crop, img, gt_3D] = get_varialbe(split, [input_2D, input_2D_crop, img, gt_3D], torch.device(0))

			N = input_2D.size(0)

			out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels)
			out_target[:, :, 14] = 0
			gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)

			if out_target.size(1) > 1:
				out_target_single = out_target[:, opt.pad].unsqueeze(1)
				gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
			else:
				out_target_single = out_target
				gt_3D_single = gt_3D

			if opt.test_augmentation and split =='test':
				input_2D, output_3D, output_3D_VTE = input_augmentation(img, input_2D.squeeze(), input_2D_crop.squeeze(), model, joints_left, joints_right)
			else:
				output_3D, output_3D_VTE = model(img, input_2D.squeeze(), input_2D_crop.squeeze())

			output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)

			output_3D_single = output_3D

			if split == 'train':
				pred_out = output_3D_single

			elif split == 'test':
				pred_out = output_3D_single

			if split == 'train':
				loss = mpjpe_cal(output_3D_single, out_target_single)
			elif split == 'test':
				loss = mpjpe_cal(pred_out, out_target) + mpjpe_cal(output_3D_single, out_target_single)

		loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

		if split == 'train':
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if not opt.MAE:

				if opt.refine:
					post_out[:,:,14,:] = 0
					joint_error = mpjpe_cal(post_out, out_target_single).item()
				else:
					pred_out[:,:,14,:] = 0
					joint_error = mpjpe_cal(pred_out, out_target_single).item()

				error_sum.update(joint_error*N, N)

		elif split == 'test':
			if opt.MAE:
				joint_error_test = mpjpe_cal(torch.cat((input_2D[:, ~mask], input_2D[:, mask]), dim=1), output_2D).item()
			else:
				pred_out[:, :, 14, :] = 0
				joint_error_test = mpjpe_cal(pred_out, out_target).item()
				out = pred_out

			if opt.train == 0:
				for seq_cnt in range(len(seq)):
					seq_name = seq[seq_cnt]
					if seq_name in data_inference:
						data_inference[seq_name] = np.concatenate(
							(data_inference[seq_name], out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
					else:
						data_inference[seq_name] = out[seq_cnt].permute(2, 1, 0).cpu().numpy()

			error_sum_test.update(joint_error_test * N, N)

	if split == 'train':
		if opt.MAE:
			return loss_all['loss'].avg*1000
		else:
			return loss_all['loss'].avg, error_sum.avg
	elif split == 'test':
		if opt.MAE:
			return error_sum_test.avg*1000
		if opt.refine:
			p1, p2 = print_error(opt.dataset, action_error_sum_post_out, opt.train)
		else:
			if opt.train == 0:
				for seq_name in data_inference.keys():
					data_inference[seq_name] = data_inference[seq_name][:, :, None, :]
				mat_path = os.path.join(opt.checkpoint, 'inference_data.mat')
				scio.savemat(mat_path, data_inference)

		return error_sum_test.avg

def input_augmentation_MAE(input_2D, model, joints_left, joints_right, mask, spatial_mask=None):
	N, _, T, J, C = input_2D.shape

	output_2D_flip = model(img[:, 1], input_2D[:, 1], input_2D_crop[:, 1])

	output_2D_flip[:, 0] *= -1

	output_2D_flip[:, :, :, joints_left + joints_right] = output_2D_flip[:, :, :, joints_right + joints_left]

	output_2D_non_flip = model(input_2D_non_flip, mask, spatial_mask)

	output_2D = (output_2D_non_flip + output_2D_flip) / 2

	input_2D = input_2D_non_flip

	return input_2D, output_2D

def input_augmentation(img, input_2D, input_2D_crop, model, joints_left, joints_right):
	output_3D_flip, _ = model(img[:, 1], input_2D[:, 1], input_2D_crop[:, 1])

	output_3D_flip[:, 0] *= -1

	output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]

	output_3D_non_flip, _ = model(img[:, 0], input_2D[:, 0], input_2D_crop[:, 0])

	output_3D = (output_3D_non_flip + output_3D_flip) / 2

	return input_2D, output_3D, None

def match_name_keywords(n, name_keywords):
	out = False
	for b in name_keywords:
		if b in n:
			out = True
			break
	return out

if __name__ == '__main__':
	opt.manualSeed = 1

	random.seed(opt.manualSeed)
	torch.manual_seed(opt.manualSeed)
	np.random.seed(opt.manualSeed)
	torch.cuda.manual_seed_all(opt.manualSeed)

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	if opt.train == 1:
		logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
							filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
			
	root_path = opt.root_path
	actions = define_actions(opt.actions)

	if opt.train:
		train_data = Fusion(opt=opt, train=True, root_path=root_path, MAE=opt.MAE)
		train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
													   shuffle=True, num_workers=int(opt.workers), pin_memory=True)
	if opt.test:
		test_data = Fusion(opt=opt, train=False, root_path=root_path, MAE=opt.MAE)
		test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
													  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

	opt.out_joints = 17

	if opt.backbone == 'hrnet_48':
		# Default setting
		pass

	elif opt.backbone == 'hrnet_32':
		# Override the default setting
		config.model.backbone.checkpoint = 'dataset/pretrained/pose_hrnet_w32_256x192.pth'
		config.model.backbone.STAGE2.NUM_CHANNELS = [32, 64]
		config.model.backbone.STAGE3.NUM_CHANNELS = [32, 64, 128]
		config.model.backbone.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
		# config.model.backbone.STAGE3.NUM_MODULES = 1
		# config.model.backbone.STAGE4.NUM_MODULES = 1
		config.model.poseformer.base_dim = 32
		config.model.poseformer.embed_dim_ratio = 64

	else:
		raise NotImplementedError("This backbone is not implemented yet.")

	model = VolumetricTriangulationNet(config)
	ret = model.backbone.load_state_dict(torch.load(config.model.backbone.checkpoint), strict=False)
	print(ret)
	print("Loading backbone from {}".format(config.model.backbone.checkpoint))

	model_params = 0
	for parameter in model.parameters():
		model_params += parameter.numel()
	print('INFO: Trainable parameter count:', model_params)

	if opt.reload == 1:
		if opt.backbone == 'hrnet_32':
			state_dict = torch.load('checkpoint/HRNet_32_64_no_refine_24_3214.pth')
		elif opt.backbone == 'hrnet_48':
			state_dict = torch.load('checkpoint/HRNet_48_96_no_refine_45_3125.pth')
		state_dict_ = {}
		for k, v in state_dict.items():
			state_dict_[k[7:]] = v
		ret = model.load_state_dict(state_dict_)
		print(ret)

	lr = opt.lr
	
	param_dicts = [
	{
		"params":
			[p for n, p in model.volume_net.named_parameters()
			if not match_name_keywords(n, 'sampling_offsets') and p.requires_grad],
		"lr": opt.lr,
	},
	{
		"params":
			[p for n, p in model.volume_net.named_parameters()
			if match_name_keywords(n, 'sampling_offsets') and p.requires_grad],
		"lr": opt.lr * 0.1,
	},
	]

	model = nn.DataParallel(model).cuda()

	optimizer_all = optim.AdamW(param_dicts, weight_decay=0.1)

	for epoch in range(1, opt.nepoch):
		if opt.train == 1:
			if not opt.MAE:
				loss, mpjpe = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
			else:
				loss = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
		if opt.test == 1:
			if not opt.MAE:
				p1 = val(opt, actions, test_dataloader, model)
			else:
				p1 = val(opt, actions, test_dataloader, model)
			data_threshold = p1

			if opt.train and data_threshold < opt.previous_best_threshold:
				print("save best checkpoint,", data_threshold)
				if opt.MAE:
					opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold,
												   model['MAE'], 'MAE')

				else:
					opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model, 'no_refine')

					if opt.refine:
						opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
															  data_threshold, model['refine'], 'refine')
				opt.previous_best_threshold = data_threshold

			if opt.train == 0:
				print('p1: %.2f' % (p1))
				break
			else:
				if opt.MAE:
					logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f' % (
					epoch, lr, loss, p1))
					print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f' % (epoch, lr, loss, p1))
				else:
					logging.info('epoch: %d, lr: %.7f, loss: %.4f, MPJPE: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))
					print('e: %d, lr: %.7f, loss: %.4f, M: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))

		if epoch % opt.large_decay_epoch == 0: 
			for param_group in optimizer_all.param_groups:
				param_group['lr'] *= opt.lr_decay_large
				lr *= opt.lr_decay_large
		else:
			for param_group in optimizer_all.param_groups:
				param_group['lr'] *= opt.lr_decay
				lr *= opt.lr_decay








