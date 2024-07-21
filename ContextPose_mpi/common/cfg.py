import yaml
from easydict import EasyDict as edict
import numpy as np
import os

config = edict()

config.title = "human36m_vol_softmax_single"
config.kind = "human36m"
config.azureroot = ""
config.logdir = "logs"
config.batch_output = False
config.vis_freq = 1000
config.vis_n_elements = 10


# model definition
config.model = edict()
config.model.name = "vol"
config.model.kind = "mpii"
config.model.image_shape = [384, 384]
config.model.heatmap_shape = [96, 96]
config.model.heatmap_softmax = True
config.model.heatmap_multiplier = 100.0
config.model.init_weights = True
config.model.checkpoint = None

config.model.backbone = edict()
config.model.backbone.name = "resnet152"
config.model.backbone.style = "simple"
config.model.backbone.num_final_layer_channel = 17
config.model.backbone.num_joints = 17
config.model.backbone.num_layers = 152
config.model.backbone.init_weights = True
config.model.backbone.fix_weights = True
# config.model.backbone.checkpoint = "dataset/pretrained/pose_hrnet_w32_256x192.pth"
config.model.backbone.checkpoint = "dataset/pretrained/pose_hrnet_w48_256x192.pth"
# config.model.backbone.checkpoint = "dataset/pretrained/CPN50_256x192.pth.tar"

config.model.backbone.NUM_JOINTS = 17
config.model.backbone.PRETRAINED_LAYERS = ['*']
config.model.backbone.STEM_INPLANES = 64
config.model.backbone.FINAL_CONV_KERNEL = 1

config.model.backbone.STAGE2 = edict()
config.model.backbone.STAGE2.NUM_MODULES = 1
config.model.backbone.STAGE2.NUM_BRANCHES = 2
config.model.backbone.STAGE2.NUM_BLOCKS = [4, 4]
config.model.backbone.STAGE2.NUM_CHANNELS = [48, 96]
config.model.backbone.STAGE2.BLOCK = 'BASIC'
config.model.backbone.STAGE2.FUSE_METHOD = 'SUM'

config.model.backbone.STAGE3 = edict()
config.model.backbone.STAGE3.NUM_MODULES = 4
config.model.backbone.STAGE3.NUM_BRANCHES = 3
config.model.backbone.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.model.backbone.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.model.backbone.STAGE3.BLOCK = 'BASIC'
config.model.backbone.STAGE3.FUSE_METHOD = 'SUM'

config.model.backbone.STAGE4 = edict()
config.model.backbone.STAGE4.NUM_MODULES = 3
config.model.backbone.STAGE4.NUM_BRANCHES = 4
config.model.backbone.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.model.backbone.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.model.backbone.STAGE4.BLOCK = 'BASIC'
config.model.backbone.STAGE4.FUSE_METHOD = 'SUM'

config.model.volume_net = edict()
config.model.volume_net.volume_aggregation_method = "softmax"
config.model.volume_net.use_gt_pelvis = False
config.model.volume_net.cuboid_size = 2500.0
config.model.volume_net.volume_size = 64
config.model.volume_net.volume_multiplier = 1.0
config.model.volume_net.volume_softmax = True
config.model.volume_net.use_feature_v2v = True
config.model.volume_net.att_channels = 51
config.model.volume_net.temperature = 1500

config.model.poseformer = edict()
config.model.poseformer.base_dim = 48
config.model.poseformer.embed_dim_ratio = 96
config.model.poseformer.depth = 4
config.model.poseformer.levels = 4

# loss related params
config.loss = edict()
config.loss.criterion = "MAE"
config.loss.mse_smooth_threshold = 0
config.loss.grad_clip = 0
config.loss.scale_keypoints_3d = 0.1
config.loss.use_volumetric_ce_loss = True
config.loss.volumetric_ce_loss_weight = 0.01
config.loss.use_global_attention_loss = True
config.loss.global_attention_loss_weight = 1000000

# dataset related params
config.dataset = edict()
config.dataset.kind = "human36m"
config.dataset.data_format = ''
config.dataset.transfer_cmu_to_human36m = False
config.dataset.root = "../H36M-Toolbox/images/"
config.dataset.extra_root = "data/human36m/extra"
config.dataset.train_labels_path = "data/human36m/extra/human36m-multiview-labels-GTbboxes.npy"
config.dataset.val_labels_path = "data/human36m/extra/human36m-multiview-labels-GTbboxes.npy"
config.dataset.train_dataset = "multiview_human36m"
config.dataset.val_dataset = "human36m"

# train related params
config.train = edict()
config.train.n_objects_per_epoch = 15000
config.train.n_epochs = 9999
config.train.n_iters_per_epoch = 5000
config.train.batch_size = 3
config.train.optimizer = 'Adam'
config.train.backbone_lr = 0.0001
config.train.backbone_lr_step = [1000]
config.train.backbone_lr_factor = 0.1
config.train.process_features_lr = 0.001
config.train.volume_net_lr = 0.001
config.train.volume_net_lr_decay = 0.99
config.train.volume_net_lr_step = [1000]
config.train.volume_net_lr_factor = 0.5
config.train.with_damaged_actions = True
config.train.undistort_images = True
config.train.scale_bbox = 1.0
config.train.ignore_cameras = []
config.train.crop = True
config.train.erase = False
config.train.shuffle = True
config.train.randomize_n_views = True
config.train.min_n_views = 1
config.train.max_n_views = 1
config.train.num_workers = 8
config.train.limb_length_path = "data/human36m/extra/mean_and_std_limb_length.h5"
config.train.pred_results_path = "data/pretrained/human36m/human36m_alg_10-04-2019/checkpoints/0060/results/train.pkl"

# val related params
config.val = edict()
config.val.flip_test = True
config.val.batch_size = 6
config.val.with_damaged_actions = True
config.val.undistort_images = True
config.val.scale_bbox = 1.0
config.val.ignore_cameras = []
config.val.crop = True
config.val.erase = False
config.val.shuffle = False
config.val.randomize_n_views = True
config.val.min_n_views = 1
config.val.max_n_views = 1
config.val.num_workers = 10
config.val.retain_every_n_frames_in_test = 1
config.val.limb_length_path = "data/human36m/extra/mean_and_std_limb_length.h5"
config.val.pred_results_path = "data/pretrained/human36m/human36m_alg_10-04-2019/checkpoints/0060/results/val.pkl"

def update_dict(v, cfg):
    for kk, vv in v.items():
        if kk in cfg:
            if isinstance(vv, dict):
                update_dict(vv, cfg[kk])
            else:
                cfg[kk] = vv
        else:
            raise ValueError("{} not exist in cfg.py".format(kk))


def update_config(path):
    exp_config = None
    with open(path) as fin:
        exp_config = edict(yaml.safe_load(fin))
        update_dict(exp_config, config)


def handle_azureroot(config_dict, azureroot):
    for key in config_dict.keys():
        if isinstance(config_dict[key], str):
            if config_dict[key].startswith('data/'):
                config_dict[key] = os.path.join(azureroot, config_dict[key])
        elif isinstance(config_dict[key], dict):
            handle_azureroot(config_dict[key], azureroot)


def update_dir(azureroot, logdir):
    config.azureroot = azureroot
    config.logdir = os.path.join(config.azureroot, logdir)
    if config.model.checkpoint != None and not config.model.checkpoint.startswith('data/'):
        config.model.checkpoint = os.path.join(config.azureroot, config.model.checkpoint)
    handle_azureroot(config, config.azureroot)   

   