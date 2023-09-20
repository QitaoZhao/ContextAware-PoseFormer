import pickle
import cv2
import numpy as np
import os.path as osp
import h5py


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

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
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


def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

# train_data = pickle.load(file=open('./h36m_train_hr.pkl', 'rb'))
# img = train_data[0]
# path = osp.join('images', img['image'])
# c = img['center'] # [x, y] 从左上角开始计算
# s = img['scale']
# print(s)
# pose2d = img['joints_2d_gt_crop']
# print(pose2d)
# print(c, s)
# pose2d_1 = np.zeros_like(pose2d)
# pose2d_1_inv = np.zeros_like(pose2d)
# pose2d_2 = np.zeros_like(pose2d)
# pose2d_2_inv = np.zeros_like(pose2d)
# box = img['box']
# left_top = np.array(pose2d[5])

# data_numpy = cv2.imread(
#                 path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
#             )

# h, w, _ = data_numpy.shape

# trans = get_affine_transform(c, s, 0, [192, 256])
# trans_inv = get_affine_transform(c, s, 0, [192, 256], inv=1)
# left_top = affine_transform(left_top, trans)
# input = cv2.warpAffine(
#             data_numpy,
#             trans,
#             ([192, 256]),
#             flags=cv2.INTER_LINEAR)

# for i in range(17):
#     pose2d_1[i] = affine_transform(pose2d[i], trans)
#     pose2d_2[i] = pose2d[i] / np.array([192, 256]) * np.array([288, 384])
#     cv2.circle(input, (int(pose2d[i,0]), int(pose2d[i,1])), 3, (0, 255, 0), -1)
# print(pose2d_1)
# print(pose2d_1/[4,4])
# retval = cv2.imwrite("./demo/demo_00.jpg", input)

# trans = get_affine_transform(c, s, 0, [288, 384])
# input = cv2.warpAffine(
#             data_numpy,
#             trans,
#             ([288, 384]),
#             flags=cv2.INTER_LINEAR)

# for i in range(17):
#     pose2d_1[i] = affine_transform(pose2d[i], trans)
#     print(pose2d_2[i])
#     cv2.circle(input, (int(pose2d_2[i,0]), int(pose2d_2[i,1])), 3, (0, 255, 0), -1)
# print(pose2d_1)
# print(pose2d_1/[4,4])
# retval = cv2.imwrite("./demo/demo_01.jpg", input)

# trans = get_affine_transform(c, s, 0, [192//4, 256//4])
# input = cv2.warpAffine(
#             data_numpy,
#             trans,
#             ([192//4, 256//4]),
#             flags=cv2.INTER_LINEAR)
# for i in range(17):
#     pose2d_2[i] = affine_transform(pose2d[i], trans)
    # cv2.circle(input, (int(pose2d_1[i,0]), int(pose2d_1[i,1])), 3, (255, 0, 0), -1)
# print(pose2d_2)
# print(pose2d_2-pose2d_1/[4,4])
# retval = cv2.imwrite("./demo/demo_01.jpg", input)


# cv2.circle(input, (int(left_top[0]), int(left_top[1])), 10, (255, 0, 0), -1)
# retval = cv2.imwrite("./demo/demo_00.jpg", input)

# input = input[:, ::-1]
# left_top = affine_transform(left_top, trans)
# cv2.circle(input, (int(w-left_top[0]-1),int(left_top[1])), 10, (255, 0, 0), -1)
# retval = cv2.imwrite(".demo/demo_new.jpg", input) # "/demo.jpg" 会保存到根目录



# joints_left = [4, 5, 6, 11, 12, 13] 
# joints_right = [1, 2, 3, 14, 15, 16]

# save_dir = osp.join('h36m_256x192', 'S1')
# action = 'act_{:02d}_subact_{:02d}'.format(2, 1)
# file = h5py.File(osp.join(save_dir, action+".h5"), "r")

# image = file['data'][:][55, 0, 0]
# pose2d = file['pose2d'][:][55, 0, 0]
# # image_flip = file['data'][:][0, 0, 1]
# image_flip = np.array(image[:, ::-1], copy=True)
# # pose2d_flip = file['pose2d'][:][-1, 0, 1]
# pose2d_flip = np.array(pose2d, copy=True)
# pose2d_flip[:, 0] = image.shape[1] - pose2d_flip[:, 0] - 1
# pose2d_flip[joints_left + joints_right] = pose2d_flip[joints_right + joints_left]
# print(image.shape, pose2d.shape)
# # print((image!=image_flip[:,::-1]).sum())

# for i in range(17):
#     cv2.circle(image, (int(pose2d[i, 0]), int(pose2d[i, 1])), 2, (255, 0, 0), -1)
#     cv2.circle(image_flip, (int(pose2d_flip[i, 0]), int(pose2d_flip[i, 1])), 2, (0, 0, 255), -1)

# img = cv2.resize(image, (192//4,256//4))
# retval = cv2.imwrite("./demo/demo_2023_small.jpg", img)
# retval = cv2.imwrite("./demo/demo_2023.jpg", image)
# retval = cv2.imwrite("./demo/demo_2023_flip.jpg", image_flip)








