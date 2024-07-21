import os
import cv2
from tqdm import tqdm
import numpy as np


def _infer_box(pose3d, camera, rootIdx):
	root_joint = pose3d[rootIdx, :]
	tl_joint = root_joint.copy()
	tl_joint[0] -= 1000.0
	tl_joint[1] -= 900.0
	br_joint = root_joint.copy()
	br_joint[0] += 1000.0
	br_joint[1] += 1100.0
	tl_joint = np.reshape(tl_joint, (1, 3))
	br_joint = np.reshape(br_joint, (1, 3))

	tl2d = _weak_project(tl_joint, camera['focal_length'][0], camera['focal_length'][1], camera['center'][0],
						 camera['center'][1]).flatten()

	br2d = _weak_project(br_joint, camera['focal_length'][0], camera['focal_length'][1], camera['center'][0],
						 camera['center'][1]).flatten()
	return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
	pose2d = pose3d[:, :2] / pose3d[:, 2:3]
	pose2d[:, 0] *= fx
	pose2d[:, 1] *= fy
	pose2d[:, 0] += cx
	pose2d[:, 1] += cy
	return pose2d


def get_3rd_point(a, b):
	direct = a - b
	return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
	sn, cs = np.sin(rot_rad), np.cos(rot_rad)

	src_result = [0, 0]
	src_result[0] = src_point[0] * cs - src_point[1] * sn
	src_result[1] = src_point[0] * sn + src_point[1] * cs

	return src_result


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


def affine_transform(pt, t):
	new_pt = np.array([pt[0], pt[1], 1.]).T
	new_pt = np.dot(t, new_pt)
	return new_pt[:2]


mpi_inf_3dhp_cameras_intrinsic_params = [
	{
		'id': 'cam_0',
		'center': [1024.704, 1051.394],
		'focal_length': [1497.693, 1497.103],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_1',
		'center': [1030.519, 1052.626],
		'focal_length': [1495.217, 1495.52],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_2',
		'center': [983.8873, 987.5902],
		'focal_length': [1495.587, 1497.828],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_3',
		'center': [1029.06, 1041.409],
		'focal_length': [1495.886, 1496.033],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': -110, # Only used for visualization
	},
	{
		'id': 'cam_4',
		'center': [987.6075, 1019.069],
		'focal_length': [1490.952, 1491.108],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_5',
		'center': [1012.331, 998.5009],
		'focal_length': [1500.414, 1499.971],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_6',
		'center': [999.7319, 1010.251],
		'focal_length': [1498.471, 1498.8],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_7',
		'center': [987.2716, 976.8773],
		'focal_length': [1498.831, 1499.674],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_8',
		'center': [1017.387, 1043.032],
		'focal_length': [1500.172, 1500.837],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_9',
		'center': [1010.423, 1037.096],
		'focal_length': [1501.554, 1501.9],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_10',
		'center': [1041.614, 997.0433],
		'focal_length': [1498.423, 1498.585],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_11',
		'center': [1009.802, 999.9984],
		'focal_length': [1495.779, 1493.703],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_12',
		'center': [1000.56, 1014.975],
		'focal_length': [1501.326, 1501.491],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'cam_13',
		'center': [1005.702, 1004.214],
		'focal_length': [1496.961, 1497.378],
		'radial_distortion': [0, 0, 0],
		'tangential_distortion': [0, 0],
		'res_w': 2048,
		'res_h': 2048,
		'azimuth': 70, # Only used for visualization
	},
	{
		'id': 'TS56',
		'center': [939.85754016, 560.140743168],
		'focal_length': [1683.98345952, 1672.59370772],
		'radial_distortion': [-0.276859611, 0.131125256, -0.049318332],
		'tangential_distortion': [-0.000360494, -0.001149441],
		'res_w': 1920,
		'res_h': 1080,
		'azimuth': 70, # Only used for visualization
	},
]


subject_list = [1, 2, 3, 4, 5, 6, 7, 8]
sequence_list = [1, 2]
camera_list = [0, 1, 2, 4, 5, 6, 7, 8]

data = np.load("dataset/data_train_3dhp.npz",allow_pickle=True)['data'].item()
for sub in subject_list:
	for seq in sequence_list:
		for cam in camera_list:
			anim = data["S{:01d} Seq{:01d}".format(sub, seq)][0][str(cam)]

			data_3d = anim['data_3d']
			data_2d = anim['data_2d']

			subdir_format = 's_{:02d}_seq_{:02d}_ca_{:02d}'
			subdir = subdir_format.format(sub, seq, cam)
			print(subdir)
			fileformat = 'dataset/mpi_inf_3dhp/images' + '/' + subdir + '/' + subdir + '_{:06d}.jpg'

			assert data_2d.shape[0] == data_3d.shape[0]
			valid_frame = data_2d.shape[0]

			for i in tqdm(range(valid_frame)):
				file_path = fileformat.format(i+1)

				data_numpy = cv2.imread(
								file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
							)

				if data_numpy.shape[0] == 256:
					continue

				box = _infer_box(data_3d[i], mpi_inf_3dhp_cameras_intrinsic_params[cam], 14)
				c = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
				s = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)

				trans = get_affine_transform(c, s, 0, [192, 256])

				data_numpy = cv2.warpAffine(
				            data_numpy,
				            trans,
				            ([192, 256]),
				            flags=cv2.INTER_LINEAR)

				os.remove(file_path)
				retval = cv2.imwrite(file_path, data_numpy)


