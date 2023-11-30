import os
import shutil
import argparse
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
np.set_printoptions(suppress=True)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn import datasets
from mvn.models.conpose import VolumetricTriangulationNet
from mvn.models.loss import MPJPE, KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss

from mvn.utils import misc
from mvn.utils.cfg import config, update_config, update_dir
from mvn.datasets import utils as dataset_utils


joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
	parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
	parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")

	parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument('--sync_bn', action='store_true', help="If set, then utilize pytorch convert_syncbn_model")

	parser.add_argument("--logdir", type=str, default="logs/", help="Path, where logs will be stored")
	parser.add_argument("--azureroot", type=str, default="", help="Root path, where codes are stored")

	parser.add_argument("--frame", type=int, default=1, help="Frame number to be used.")

	args = parser.parse_args()
	# update config
	update_config(args.config)
	update_dir(args.azureroot, args.logdir)
	return args


def setup_human36m_dataloaders(config, is_train, distributed_train, rank = None, world_size = None):
	train_dataloader = None
	if is_train:
		# train
		train_dataset = eval('datasets.' + config.dataset.train_dataset)(
			root=config.dataset.root,
			pred_results_path=config.train.pred_results_path,
			train=True,
			test=False,
			image_shape=config.model.image_shape,
			labels_path=config.dataset.train_labels_path,
			with_damaged_actions=config.train.with_damaged_actions,
			scale_bbox=config.train.scale_bbox,
			kind=config.kind,
			undistort_images=config.train.undistort_images,
			ignore_cameras=config.train.ignore_cameras,
			crop=config.train.crop,
			erase=config.train.erase,
			data_format=config.dataset.data_format,
			frame=args.frame
		)

		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

		train_dataloader = DataLoader(
			train_dataset,
			batch_size=config.train.batch_size,
			shuffle=config.train.shuffle and (train_sampler is None), # debatable
			sampler=train_sampler,
			num_workers=config.train.num_workers,
			worker_init_fn=dataset_utils.worker_init_fn,
			pin_memory=True
		)

	# val
	val_dataset = eval('datasets.' + config.dataset.val_dataset)(
		root=config.dataset.root,
		pred_results_path=config.val.pred_results_path,
		train=False,
		test=True,
		image_shape=config.model.image_shape,
		labels_path=config.dataset.val_labels_path,
		with_damaged_actions=config.val.with_damaged_actions,
		retain_every_n_frames_in_test=config.val.retain_every_n_frames_in_test,
		scale_bbox=config.val.scale_bbox,
		kind=config.kind,
		undistort_images=config.val.undistort_images,
		ignore_cameras=config.val.ignore_cameras,
		crop=config.val.crop,
		erase=config.val.erase,
		rank=rank,
		world_size=world_size,
		data_format=config.dataset.data_format,
		frame=args.frame
	)

	val_dataloader = DataLoader(
		val_dataset,
		batch_size=config.val.batch_size,
		shuffle=config.val.shuffle,
		# collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.val.randomize_n_views,
		#                                          min_n_views=config.val.min_n_views,
		#                                          max_n_views=config.val.max_n_views),
		num_workers=config.val.num_workers,
		worker_init_fn=dataset_utils.worker_init_fn,
		pin_memory=True
	)

	return train_dataloader, val_dataloader, train_sampler, val_dataset.dist_size


def setup_dataloaders(config, is_train=True, distributed_train=False, rank = None, world_size=None):
	if config.dataset.kind == 'human36m':
		train_dataloader, val_dataloader, train_sampler, dist_size = setup_human36m_dataloaders(config, is_train, distributed_train, rank, world_size)
		_, whole_val_dataloader, _, _ = setup_human36m_dataloaders(config, is_train, distributed_train)
	else:
		raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))
	
	return train_dataloader, val_dataloader, train_sampler, whole_val_dataloader, dist_size


def setup_experiment(config, model_name, is_train=True):
	prefix = "" if is_train else "eval_"

	if config.title:
		experiment_title = config.title + "_" + model_name
	else:
		experiment_title = model_name

	experiment_title = "ConPose"
	experiment_title = prefix + experiment_title

	experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
	print("Experiment name: {}".format(experiment_name))

	experiment_dir = os.path.join(config.logdir, experiment_name)
	os.makedirs(experiment_dir, exist_ok=True)

	checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
	os.makedirs(checkpoints_dir, exist_ok=True)

	shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

	# tensorboard
	writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

	# dump config to tensorboard
	writer.add_text(misc.config_to_str(config), "config", 0)

	return experiment_dir, writer


def one_epoch_full(model, criterion, optimizer, config, dataloader, device, epoch, n_iters_total=0, is_train=True, lr=None, mean_and_std=None, limb_length=None, caption='', master=False, experiment_dir=None, writer=None, whole_val_dataloader=None, dist_size=None):
	name = "train" if is_train else "val"
	model_type = config.model.name

	if is_train:
		epoch_loss_3d_train = 0
		N = 0

		if config.model.backbone.fix_weights:
			model.module.backbone.eval()
			model.module.volume_net.train()
		else:
			model.train()
	else:
		model.eval()

	metric_dict = defaultdict(list)

	results = defaultdict(list)

	# used to turn on/off gradients
	grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
	with grad_context():
		prefetcher = dataset_utils.data_prefetcher(dataloader, device, is_train, config.val.flip_test)

		# for iter_i, batch in iterator:
		batch = prefetcher.next()
		while batch is not None:
			# measure data loading time
			end = time.time()
			data_time = time.time() - end

			images_batch, keypoints_3d_gt, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop = batch

			if model_type == "vol":
				if (not is_train) and config.val.flip_test:
					keypoints_3d_pred = model(images_batch[:, 0], keypoints_2d_batch_cpn[:, 0], keypoints_2d_batch_cpn_crop[:, 0].clone())
					keypoints_3d_pred_flip = model(images_batch[:, 1], keypoints_2d_batch_cpn[:, 1], keypoints_2d_batch_cpn_crop[:, 1].clone())
					keypoints_3d_pred_flip[:, :, :, 0] *= -1
					keypoints_3d_pred_flip[:, :, joints_left + joints_right] = keypoints_3d_pred_flip[:, :, joints_right + joints_left]
					keypoints_3d_pred = torch.mean(torch.cat((keypoints_3d_pred, keypoints_3d_pred_flip), dim=1), dim=1,
												   keepdim=True)
					del keypoints_3d_pred_flip


				else:    
					keypoints_3d_pred = model(images_batch, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop)

			n_joints = keypoints_3d_pred.shape[1]


			# calculate loss
			total_loss = 0.0
			loss = criterion(keypoints_3d_pred, keypoints_3d_gt)

			total_loss += loss
			metric_dict[config.loss.criterion].append(loss.item())

			metric_dict['total_loss'].append(total_loss.item())

			if is_train:
				epoch_loss_3d_train += keypoints_3d_gt.shape[0] * loss.item()
				N += keypoints_3d_gt.shape[0]

				if not torch.isnan(total_loss):
					# for key in opt_dict.keys():
					#     opt_dict[key].zero_grad()
					optimizer.zero_grad()
					total_loss.backward()

					if config.loss.grad_clip:
						torch.nn.utils.clip_grad_norm_(model.parameters(), config.loss.grad_clip / config.train.volume_net_lr)

					optimizer.step()

			# save answers for evalulation
			if not is_train:
				results['keypoints_gt'].append(keypoints_3d_gt.detach())    # (b, 17, 3)
				results['keypoints_3d'].append(keypoints_3d_pred.detach())    # (b, 17, 3)

			batch = prefetcher.next()

	if is_train:
		return epoch_loss_3d_train / N

	# calculate evaluation metrics
	if not is_train:
		if dist_size is not None:
			# term_list = ['keypoints_gt', 'keypoints_3d', 'proj_matricies_batch', 'indexes']
			term_list = ['keypoints_gt', 'keypoints_3d']
			for term in term_list:
				# results[term] = np.concatenate(results[term])
				results[term] = torch.cat(results[term])
				buffer = [torch.zeros(dist_size[-1], *results[term].shape[1:]).cuda() for i in range(len(dist_size))]
				scatter_tensor = torch.zeros_like(buffer[0])
				scatter_tensor[:results[term].shape[0]] = results[term]
				# scatter_tensor[:results[term].shape[0]] = torch.tensor(results[term]).cuda()
				torch.distributed.all_gather(buffer, scatter_tensor)
				results[term] = torch.cat([tensor[:n] for tensor, n in zip(buffer, dist_size)], dim = 0)#.cpu().numpy()

	if master:
		if not is_train:
			if dist_size is None:
				print('evaluating....')
				result = dataloader.dataset.evaluate(results['keypoints_gt'], results['keypoints_3d'], None, config)
			else:
				result = whole_val_dataloader.dataset.evaluate(results['keypoints_gt'], results['keypoints_3d'], None, config)

			return result


def init_distributed(args):
	if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
		return False

	torch.cuda.set_device(args.local_rank)

	assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
	assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

	os.environ['PYTHONHASHSEED'] = str(args.seed)
	torch.manual_seed(args.seed)
	torch.distributed.init_process_group(backend="nccl", init_method="env://")

	return True


def match_name_keywords(n, name_keywords):
	out = False
	for b in name_keywords:
		if b in n:
			out = True
			break
	return out


def main(args):
	is_distributed = init_distributed(args)

	master = True
	if is_distributed and os.environ["RANK"]:
		master = int(os.environ["RANK"]) == 0
		rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
	else:
		rank = world_size = None

	if is_distributed:
		device = torch.device(args.local_rank)
	else:
		device = torch.device(0)

	# config.train.n_iters_per_epoch = config.train.n_objects_per_epoch // config.train.batch_size   
	config.train.n_iters_per_epoch = None                     

	model = {
		"vol": VolumetricTriangulationNet
	}[config.model.name](config, device)

	# experiment
	experiment_dir, writer = None, None
	if master:
		experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)
		shutil.copy('mvn/models/conpose.py', experiment_dir)
		shutil.copy('mvn/models/pose_dformer.py', experiment_dir)
		shutil.copy('train.py', experiment_dir)
		# shutil.copy('mvn/models/pose_cformer.py', experiment_dir)

	# logger = Logger(log_path=experiment_dir).logger
	# logger.parent = None
	print("args: {}".format(args))
	print("Number of available GPUs: {}".format(torch.cuda.device_count()))

	if config.model.init_weights:
		checkpoint_path = None
		if config.model.checkpoint != None:
			checkpoint_path = config.model.checkpoint
		elif os.path.isfile(os.path.join(config.logdir, "resume_weights_path.pth")):
			checkpoint_path = torch.load(os.path.join(config.logdir, "resume_weights_path.pth"))
		if checkpoint_path != None and os.path.isfile(checkpoint_path):
			state_dict = torch.load(checkpoint_path, map_location=device)
			for key in list(state_dict.keys()):
				new_key = key.replace("module.", "")
				state_dict[new_key] = state_dict.pop(key)
			try:
				model.load_state_dict(state_dict, strict=True)
			except:
				print('Warning: Final layer do not match!')
				for key in list(state_dict.keys()):
					if 'final_layer' in key:
						state_dict.pop(key)
				model.load_state_dict(state_dict, strict=True)
			print("Successfully loaded weights for {} model from {}".format(config.model.name, checkpoint_path))
			del state_dict
		else:
			print("Failed loading weights for {} model as no checkpoint found at {}".format(config.model.name, checkpoint_path))

	elif config.model.backbone.init_weights:
		# Load HRNet
		ret = model.backbone.load_state_dict(torch.load(config.model.backbone.checkpoint), strict=False)
		# Load CPN
		# state_dict = torch.load(config.model.backbone.checkpoint, map_location=device)['state_dict']
		# for key in list(state_dict.keys()):
		# 	new_key = key.replace("module.", "")
		# 	state_dict[new_key] = state_dict.pop(key)
		# ret = model.backbone.load_state_dict(state_dict, strict=True)
		print(ret)
		print("Loading backbone from {}".format(config.model.backbone.checkpoint))

	if args.eval:
		checkpoint = torch.load("checkpoint/best_epoch.bin")['model']
		for key in list(checkpoint.keys()):
			new_key = key.replace("module.", "")
			checkpoint[new_key] = checkpoint.pop(key)

		ret = model.load_state_dict(checkpoint, strict=False)
		print(ret)
		# model.load_state_dict(checkpoint['model'], strict=True)
		print("Loading checkpoint from {}".format("checkpoint/best_epoch.bin"))

	# sync bn in multi-gpus
	if args.sync_bn:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model = model.to(device)

	# criterion
	criterion_class = {
		"MPJPE": MPJPE,
		"MSE": KeypointsMSELoss,
		"MSESmooth": KeypointsMSESmoothLoss,
		"MAE": KeypointsMAELoss
	}[config.loss.criterion]

	if config.loss.criterion == "MSESmooth":
		criterion = criterion_class(config.loss.mse_smooth_threshold).to(device)
	else:
		criterion = criterion_class().to(device)

	# optimizer
	opt_dict = None
	lr_schd_dict = None
	lr_dict = None
	lr = config.train.volume_net_lr
	lr_decay = config.train.volume_net_lr_decay
	if not args.eval:
		if config.model.name == "vol":
			opt_dict = {}
			lr_schd_dict = {}
			lr_dict = {}
			
			param_dicts = [
			# {
			# 	"params":
			# 		[p for n, p in model.backbone.named_parameters() if p.requires_grad],
			# 	"lr": config.train.volume_net_lr * 0.1,
			# },
			{
				"params":
					[p for n, p in model.volume_net.named_parameters()
					if not match_name_keywords(n, 'sampling_offsets') and p.requires_grad],
				"lr": config.train.volume_net_lr,
			},
			{
				"params":
					[p for n, p in model.volume_net.named_parameters()
					if match_name_keywords(n, 'sampling_offsets') and p.requires_grad],
				"lr": config.train.volume_net_lr * 0.1,
			},
			]
			optimizer = optim.AdamW(param_dicts, weight_decay=0.1)
		else:
			assert 0, "Only support vol optimizer."
		# load optimizer if has
		if config.model.init_weights and checkpoint_path != None:
			optimizer_path = checkpoint_path.replace('weights', 'optimizer')
			if os.path.isfile(optimizer_path):
				try:
					optimizer_dict = torch.load(optimizer_path, map_location=device)
					if config.model.name == 'vol':
						opt_dict['3d'].load_state_dict(optimizer_dict['optimizer_3d'])
						lr_schd_dict['3d'].load_state_dict(optimizer_dict['scheduler_3d'])
						if 'scheduler_2d' in optimizer_dict.keys():
							opt_dict['2d'].load_state_dict(optimizer_dict['optimizer_2d'])
							lr_schd_dict['2d'].load_state_dict(optimizer_dict['scheduler_2d'])
					else:
						assert 0, "Only support vol optimizer."
					del optimizer_dict
					print("Successfully loaded optimizer parameters for {} model".format(config.model.name))
				except:
					print("Warning: optimizer does not match! Failed loading optimizer parameters for {} model".format(config.model.name))
			else:
				print("Failed loading optimizer parameters for {} model as no optimizer found at {}".format(config.model.name, optimizer_path))

	# datasets
	if master:
		print("Loading data...")
	train_dataloader, val_dataloader, train_sampler, whole_val_dataloader, dist_size = setup_dataloaders(config, distributed_train=is_distributed, rank=rank, world_size=world_size)

	if master:
		model_params = 0
		for parameter in model.volume_net.parameters():
			model_params += parameter.numel()
		print("Trainable parameter count: " + str(model_params))

	# multi-gpu
	if is_distributed:
		model = DistributedDataParallel(model, device_ids=[device], output_device=args.local_rank) # , find_unused_parameters=True

	if not args.eval:
		# train loop
		losses_3d_train = []
		min_loss = 100000
		n_iters_total_train, n_iters_total_val = 0, 0
		
		for epoch in range(config.train.n_epochs):
			errors_p1 = []
			errors_p2 = []
			# errors_p3 = []
			start_time = time.time()
			if train_sampler is not None:
				train_sampler.set_epoch(epoch)

			if config.model.name == 'vol':
				epoch_loss_3d_train = one_epoch_full(model, criterion, optimizer, config, train_dataloader, device, epoch, n_iters_total=n_iters_total_train, is_train=True, lr=lr_dict, master=master, experiment_dir=experiment_dir, writer=writer)
				result = one_epoch_full(model, criterion, optimizer, config, val_dataloader, device, epoch, n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, whole_val_dataloader=whole_val_dataloader, dist_size=dist_size)

				losses_3d_train.append(epoch_loss_3d_train)

				if master:
					for k in result.keys():
						errors_p1.append(result[k]['MPJPE'] * 1000)
						errors_p2.append(result[k]['P_MPJPE'] * 1000)
						# errors_p3.append(result[k]['N_MPJPE'] * 1000)

					error_p1 = round(np.mean(errors_p1), 1)
					error_p2 = round(np.mean(errors_p2), 1)
					# error_p3 = round(np.mean(errors_p3), 1)                

				# for key in lr_schd_dict.keys():
				#     lr_schd_dict[key].step()
				#     try:
				#         lr_dict[key] = lr_schd_dict[key].get_last_lr()[0]
				#     except: # old PyTorch
				#         lr_dict[key] = lr_schd_dict[key].get_lr()[0]
			else:
				assert 0, "only support training vol model."

			if master:
				print('[%d] time %.2f lr %f 3d_train %f 3d_test_p1 %f 3d_test_p2 %f' % (
					epoch + 1,
					(time.time()-start_time) / 60.,
					lr,
					losses_3d_train[-1] * 1000,
					error_p1,
					error_p2))

				if error_p1 < min_loss:
					min_loss = error_p1
					print("save best checkpoint")
					torch.save({
						'epoch': epoch + 1,
						'lr': lr,
						# 'random_state': train_generator.random_state(),
						'optimizer': optimizer.state_dict(),
						'model': model.state_dict(),
					}, os.path.join(experiment_dir, "checkpoints/best_epoch.bin"))

			lr *= lr_decay
			for param_group in optimizer.param_groups:
				param_group['lr'] *= lr_decay

	else:
		errors_p1 = []
		errors_p2 = []
		errors_vel = []
		dataloader = train_dataloader if args.eval_dataset == 'train' else val_dataloader

		result = one_epoch_full(model, criterion, None, config, val_dataloader, device, None, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, whole_val_dataloader=whole_val_dataloader, dist_size=dist_size)

		if master:
			for k in result.keys():
				print(k, "p1:", result[k]['MPJPE'] * 1000, "p2:", result[k]['P_MPJPE'] * 1000, "e_vel:", result[k]['MPJVE'] * 1000)
				errors_p1.append(result[k]['MPJPE'] * 1000)
				errors_p2.append(result[k]['P_MPJPE'] * 1000)
				# errors_p3.append(result[k]['N_MPJPE'] * 1000)
				errors_vel.append(result[k]['MPJVE'] * 1000)

			error_p1 = round(np.mean(errors_p1), 1)
			error_p2 = round(np.mean(errors_p2), 1)
			error_vel = round(np.mean(errors_vel), 2)
			print("avg", "p1:", error_p1, "p2:", error_p2, "MPJVE:", error_vel)

			print("Done.")

if __name__ == '__main__':
	args = parse_args()
	main(args)
