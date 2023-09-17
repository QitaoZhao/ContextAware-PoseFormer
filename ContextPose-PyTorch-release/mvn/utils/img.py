import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def get_3rd_point(a, b):
	direct = a - b
	return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
		center, scale, rot, output_size,
		shift=np.array([0, 0], dtype=np.float32), inv=0
):
	center = np.array(center)
	scale = np.array(scale)

	scale_tmp = scale * 200.0
	src_w = scale_tmp[0]
	dst_w = output_size[0]
	dst_h = output_size[1]

	# rot_rad = np.pi * rot / 180

	# src_dir = get_dir([0, (src_w-1) * -0.5], rot_rad)
	src_dir = np.array([0, (src_w-1) * -0.5], np.float32)
	dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
	src = np.zeros((3, 2), dtype=np.float32)
	dst = np.zeros((3, 2), dtype=np.float32)
	src[0, :] = center + scale_tmp * shift
	src[1, :] = center + src_dir + scale_tmp * shift
	dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
	dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

	src[2:, :] = get_3rd_point(src[0, :], src[1, :])
	dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

	if inv:
		trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
	else:
		trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

	return trans


def crop_image(image, center, scale, output_size):
	"""Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
	Args:
		image numpy array of shape (height, width, 3): input image
		bbox tuple of size 4: input bbox (left, upper, right, lower)

	Returns:
		cropped_image numpy array of shape (height, width, 3): resulting cropped image

	"""

	trans = get_affine_transform(center, scale, 0, output_size)
	image = cv2.warpAffine(
		image,
		trans,
		(output_size),
		flags=cv2.INTER_LINEAR)

	return image
	

# def crop_image(image, bbox):
#     """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
#     Args:
#         image numpy array of shape (height, width, 3): input image
#         bbox tuple of size 4: input bbox (left, upper, right, lower)

#     Returns:
#         cropped_image numpy array of shape (height, width, 3): resulting cropped image

#     """

#     image_pil = Image.fromarray(image)
#     image_pil = image_pil.crop(bbox)

#     return np.asarray(image_pil)


def resize_image(image, shape):
	return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def get_square_bbox(bbox):
	"""Makes square bbox from any bbox by stretching of minimal length side

	Args:
		bbox tuple of size 4: input bbox (left, upper, right, lower)

	Returns:
		bbox: tuple of size 4:  resulting square bbox (left, upper, right, lower)
	"""

	left, upper, right, lower = bbox
	width, height = right - left, lower - upper

	if width > height:
		y_center = (upper + lower) // 2
		upper = y_center - width // 2
		lower = upper + width
	else:
		x_center = (left + right) // 2
		left = x_center - height // 2
		right = left + height

	return left, upper, right, lower


def scale_bbox(bbox, scale):
	left, upper, right, lower = bbox
	width, height = right - left, lower - upper

	x_center, y_center = (right + left) // 2, (lower + upper) // 2
	new_width, new_height = int(scale * width), int(scale * height)

	new_left = x_center - new_width // 2
	new_right = new_left + new_width

	new_upper = y_center - new_height // 2
	new_lower = new_upper + new_height

	return new_left, new_upper, new_right, new_lower


def to_numpy(tensor):
	if torch.is_tensor(tensor):
		return tensor.cpu().detach().numpy()
	elif type(tensor).__module__ != 'numpy':
		raise ValueError("Cannot convert {} to numpy array"
						 .format(type(tensor)))
	return tensor


def to_torch(ndarray):
	if type(ndarray).__module__ == 'numpy':
		return torch.from_numpy(ndarray)
	elif not torch.is_tensor(ndarray):
		raise ValueError("Cannot convert {} to torch tensor"
						 .format(type(ndarray)))
	return ndarray


def image_batch_to_numpy(image_batch):
	image_batch = to_numpy(image_batch)
	image_batch = np.transpose(image_batch, (0, 2, 3, 1)) # BxCxHxW -> BxHxWxC
	return image_batch


def image_batch_to_torch(image_batch):
	image_batch = np.transpose(image_batch, (0, 3, 1, 2)) # BxHxWxC -> BxCxHxW
	image_batch = to_torch(image_batch).float()
	return image_batch


def normalize_image(image):
	"""Normalizes image using ImageNet mean and std

	Args:
		image numpy array of shape (h, w, 3): image

	Returns normalized_image numpy array of shape (h, w, 3): normalized image
	"""
	return (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_image(image):
	"""Reverse to normalize_image() function"""
	return np.clip(255.0 * (image * IMAGENET_STD + IMAGENET_MEAN), 0, 255)

def erase_image(image, center_list, size = 70, use_mean = True):
	"""erase image

	Args:
		image numpy array of shape (h, w, 3): image
	"""
	new_image = image.copy()
	height, width = image.shape[:2]
	for x_center, y_center in center_list:
		x_center, y_center = int(x_center), int(y_center)
		if x_center < 0 or y_center < 0 or x_center >= width or y_center >= height:
			continue
		left, right = max(0, x_center - size // 2), min(width, x_center + size // 2 + 1)
		upper, lower = max(0, y_center - size // 2), min(height, y_center + size // 2 + 1)
		if use_mean:
			mean = image[upper:lower, left:right].mean(axis = (0, 1), keepdims = True)
			new_image[upper:lower, left:right] = mean
		else:
			new_image[upper:lower, left:right] = 0
	return new_image

def gamma_trans(img, gamma):#gamma大于1时图片变暗，小于1图片变亮
	#具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
	gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	#实现映射用的是Opencv的查表函数
	return cv2.LUT(img,gamma_table)


def draw_pic(images_batch_copy, keypoints_2d_cpn, keypoints_2d_gt, list_offsets, list_weights, index):
	b, h, w, c = images_batch_copy.shape
	# b, l, p, num_heads, num_samples, 1
	# offsets = list_offsets[-1]
	for i in tqdm(range(b)):
		if index + i <= 2000 or index + i >= 2100:
			continue
		data = images_batch_copy[i]
		data = gamma_trans(data, 2)
		kp_2d_cpn = keypoints_2d_cpn[i]
		kp_2d_gt = keypoints_2d_gt[i]
		distance = (kp_2d_cpn - kp_2d_gt)**2
		distance = np.sqrt(distance.sum(axis=-1))
		idx = distance.argmax()
		# colors = np.array([255, 255, 255])
		# for j in range(17):
		cv2.circle(data, (int(keypoints_2d_cpn[i,idx,0]), int(keypoints_2d_cpn[i,idx,1])), 3, (255, 0, 0), -1)
		cv2.circle(data, (int(keypoints_2d_gt[i,idx,0]), int(keypoints_2d_gt[i,idx,1])), 3, (0, 255, 0), -1)
		# cv2.drawMarker(data, (int(keypoints_2d_gt[i,idx,0]), int(keypoints_2d_gt[i,idx,1])), (88, 236, 246), markerType=cv2.MARKER_STAR, markerSize=3)
		for j in range(len(list_offsets)):
			offsets = list_offsets[j]
			weights = list_weights[j][i, 0, idx, :, 0]
			weights = np.lexsort((np.arange(offsets.shape[3]), weights.argsort()))
			weights = weights / offsets.shape[3]
			# weights = weights[i, 0, idx, :, 0].argmin() + 1 / offsets.shape[3]
			for k in range(offsets.shape[3]):
				x = offsets[i, 0, idx, k, 0] #+ keypoints_2d_cpn[i, idx, 0]
				y = offsets[i, 0, idx, k, 1] #+ keypoints_2d_cpn[i, idx, 1]
				color = np.array([255, 255, 255])
				color = color * (weights[k]**2)
				color = tuple ([int(x) for x in color])
				cv2.circle(data, (int(x), int(y)), 1, color, -1)
	
		# retval = cv2.imwrite("demo/demo_{:06d}.jpg".format(i+index), data)
		RGBimage = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
		PILimage = Image.fromarray(RGBimage)
		PILimage.save("new/demo_{:06d}.png".format(i+index), dpi=(200,200))
		# retval = cv2.imwrite("new/demo_{:06d}.png".format(i+index), data)

	return None





