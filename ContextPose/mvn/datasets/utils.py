import numpy as np
import torch

from mvn.utils.img import image_batch_to_torch

import os
import zipfile
import cv2
import random


joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]

class data_prefetcher():
    def __init__(self, loader, device, is_train, flip_test, backbone):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.is_train = is_train
        self.flip_test = flip_test
        self.backbone = backbone

        if backbone in ['hrnet_32', 'hrnet_48']:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().to(device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().to(device)
        elif backbone == 'cpn':
            self.mean = torch.tensor([122.7717, 115.9465, 102.9801]).cuda().to(device).view(1, 1, 1, 3)
            self.mean /= 255.

        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_batch)):
                self.next_batch[i] = self.next_batch[i].cuda(non_blocking=True).to(self.device)

            images_batch, keypoints_3d_gt, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop = self.next_batch

            images_batch = torch.flip(images_batch, [-1])

            if self.backbone in ['hrnet_32', 'hrnet_48']:
                images_batch = (images_batch / 255.0 - self.mean) / self.std
            elif self.backbone == 'cpn':
                images_batch = images_batch / 255.0 - self.mean  # for CPN
                
            keypoints_3d_gt[:, :, 1:] -= keypoints_3d_gt[:, :, :1]
            keypoints_3d_gt[:, :, 0] = 0

            if random.random() <= 0.5 and self.is_train:
                images_batch = torch.flip(images_batch, [-2])

                keypoints_2d_batch_cpn[..., 0] *= -1
                keypoints_2d_batch_cpn[..., joints_left + joints_right, :] = keypoints_2d_batch_cpn[..., joints_right + joints_left, :]

                keypoints_2d_batch_cpn_crop[:, :, 0] = 192 - keypoints_2d_batch_cpn_crop[:, :, 0] - 1
                keypoints_2d_batch_cpn_crop[:, joints_left + joints_right] = keypoints_2d_batch_cpn_crop[:, joints_right + joints_left]

                keypoints_3d_gt[:, :, :, 0] *= -1
                keypoints_3d_gt[:, :, joints_left + joints_right] = keypoints_3d_gt[:, :, joints_right + joints_left]

            if (not self.is_train) and self.flip_test:
                images_batch = torch.stack([images_batch, torch.flip(images_batch,[2])], dim=1)

                keypoints_2d_batch_cpn_flip = keypoints_2d_batch_cpn.clone()
                keypoints_2d_batch_cpn_flip[..., 0] *= -1
                keypoints_2d_batch_cpn_flip[..., joints_left + joints_right, :] = keypoints_2d_batch_cpn_flip[..., joints_right + joints_left, :]
                keypoints_2d_batch_cpn = torch.stack([keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_flip], dim=1)

                keypoints_2d_batch_cpn_crop_flip = keypoints_2d_batch_cpn_crop.clone()
                keypoints_2d_batch_cpn_crop_flip[:, :, 0] = 192 - keypoints_2d_batch_cpn_crop_flip[:, :, 0] - 1
                keypoints_2d_batch_cpn_crop_flip[:, joints_left + joints_right] = keypoints_2d_batch_cpn_crop_flip[:, joints_right + joints_left]
                keypoints_2d_batch_cpn_crop = torch.stack([keypoints_2d_batch_cpn_crop, keypoints_2d_batch_cpn_crop_flip], dim=1)

                del keypoints_2d_batch_cpn_flip, keypoints_2d_batch_cpn_crop_flip

            self.next_batch = [images_batch.float(), keypoints_3d_gt.float(), keypoints_2d_batch_cpn.float(), keypoints_2d_batch_cpn_crop.float()]


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch


def make_collate_fn(randomize_n_views=True, min_n_views=10, max_n_views=31):

    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()
        total_n_views = min(len(item['images']) for item in items)

        indexes = np.arange(total_n_views)
        if randomize_n_views:
            n_views = np.random.randint(min_n_views, min(total_n_views, max_n_views) + 1)
            indexes = np.random.choice(np.arange(total_n_views), size=n_views, replace=False)
        else:
            indexes = np.arange(total_n_views)

        batch['images'] = np.stack([np.stack([item['images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
        # batch['detections'] = np.array([[item['detections'][i] for item in items] for i in indexes]).swapaxes(0, 1)
        # batch['cameras'] = [[item['cameras'][i] for item in items] for i in indexes]

        batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
        batch['keypoints_2d_cpn'] = [item['keypoints_2d_cpn'] for item in items]
        batch['keypoints_2d_cpn_crop'] = [item['keypoints_2d_cpn_crop'] for item in items]
        # batch['cuboids'] = [item['cuboids'] for item in items]
        batch['indexes'] = [item['indexes'] for item in items]
        batch['subject'] = [item['subject'] for item in items]

        try:
            batch['pred_keypoints_3d'] = np.array([item['pred_keypoints_3d'] for item in items])
        except:
            pass

        return batch

    return collate_fn


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prepare_batch(batch, device, config):
    # images
    images_batch = []
    for image_batch in batch['images']:
        image_batch = image_batch_to_torch(image_batch)
        image_batch = image_batch.to(device)
        images_batch.append(image_batch)

    images_batch = torch.stack(images_batch, dim=0)

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, :3]).float().to(device)      # (b, n_joints, 3)

    # 2D keypoints
    keypoints_2d_batch_cpn = torch.from_numpy(np.stack(batch['keypoints_2d_cpn'], axis=0)[:, :, :2]).float().to(device)      # (b, n_joints, 3)
    keypoints_2d_batch_cpn_crop = torch.from_numpy(np.stack(batch['keypoints_2d_cpn_crop'], axis=0)[:, :, :2]).float().to(device)      # (b, n_joints, 3)

    return images_batch, keypoints_3d_batch_gt, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop

_im_zfile = []


def zipreader_imread(filename, flags=cv2.IMREAD_COLOR):
    global _im_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'" % (path))
        assert 0
    path_zip = path[0:pos_at]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found" % (path_zip))
        assert 0
    for i in range(len(_im_zfile)):
        if _im_zfile[i]['path'] == path_zip:
            path_img = os.path.join(_im_zfile[i]['zipfile'].namelist()[0], path[pos_at+2:])
            data = _im_zfile[i]['zipfile'].read(path_img)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    _im_zfile.append({
        'path': path_zip,
        'zipfile': zipfile.ZipFile(path_zip, 'r')
    })
    path_img = os.path.join(_im_zfile[-1]['zipfile'].namelist()[0], path[pos_at+2:])
    data = _im_zfile[-1]['zipfile'].read(path_img)

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)