# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates
       
mpi_inf_3dhp_skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
       joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
       joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
       
subjects_train = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
subjects_test1 = ['TS1', 'TS2', 'TS3', 'TS4']
subjects_test2 = ['TS5', 'TS6']
       
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

mpi_inf_3dhp_cameras_extrinsic_params = {
    'Train': [
        {
            'orientation': [0.9910573, 0.0000989, 0.1322565, -0.017709],
            'translation': [-562.8666, 1398.138, 3852.623],
        },
        {
            'orientation': [0.8882246, -0.0698901, 0.4388433, -0.1165721],
            'translation': [-1429.856, 738.1779, 4897.966],
        },
        {
            'orientation': [0.5651277, -0.0301201, 0.824319, -0.0148915],
            'translation': [57.25702, 1307.287, 2799.822],
        },
        {
            'orientation': [0.6670245, -0.1827152, 0.7089925, -0.1379241],
            'translation': [-284.8168, 807.9184, 3177.16],
        },
        {
            'orientation': [0.8273998, 0.0263385, 0.5589656, -0.0476783],
            'translation': [-1563.911, 801.9608, 3517.316],
        },
        {
            'orientation': [-0.568842, 0.0159665, 0.8220693, -0.0191314],
            'translation': [358.4134, 994.5658, 3439.832],
        },
        {
            'orientation': [0.2030824, -0.2818073, 0.9370704, -0.0352313],
            'translation': [569.4388, 528.871, 3687.369],
        },
        {
            'orientation': [0.00086, 0.0123344, 0.9998223, -0.0142292],
            'translation': [1378.866, 1270.781, 2631.567],
        },
        {
            'orientation': [0.7053718, 0.095632, -0.7004048, -0.0523286],
            'translation': [221.3543, 659.87, 3644.688],
        },
        {
            'orientation': [0.6914033, 0.2036966, -0.6615312, -0.2069921],
            'translation': [388.6217, 137.5452, 4216.635],
        },
        {
            'orientation': [-0.2266321, -0.2540748, 0.9401911, -0.0111636],
            'translation': [1167.962, 617.6362, 4472.351],
        },
        {
            'orientation': [-0.4536946, -0.2035304, -0.0072578, 0.8675736],
            'translation': [134.8272, 251.5094, 4570.244],
        },
        {
            'orientation': [-0.0778876, 0.8469901, -0.4230185, 0.3124046],
            'translation': [412.4695, 532.7588, 4887.095],
        },
        {
            'orientation': [0.098712, 0.8023286, -0.5397436, -0.2349501],
            'translation': [867.1278, 827.4572, 3985.159],
        },
    ],
    'chestHeight': [
        {
            'orientation': [0.7053718, 0.095632, -0.7004048, -0.0523286],
            'translation': [221.3543, 659.87, 3644.688],
        },
    ],
}


class MpiInf3dhpDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
        super().__init__(fps=25, skeleton=mpi_inf_3dhp_skeleton)
        
        self._cameras = {}
        
        for subject in subjects_train:
            self._cameras[subject] = copy.deepcopy(mpi_inf_3dhp_cameras_extrinsic_params['Train'])
            
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(mpi_inf_3dhp_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')
                
                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length']/cam['res_w']*2
                if 'translation' in cam:
                    cam['translation'] = cam['translation']/1000 # mm to meters
                
                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))
        
        for subject in subjects_test1:
            self._cameras[subject] = copy.deepcopy(mpi_inf_3dhp_cameras_extrinsic_params['chestHeight'])
            cam = self._cameras[subject] [0]
            cam.update(mpi_inf_3dhp_cameras_intrinsic_params[8])
            for k, v in cam.items():
                if k not in ['id', 'res_w', 'res_h']:
                    cam[k] = np.array(v, dtype='float32')
            
            # Normalize camera frame
            cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
            cam['focal_length'] = cam['focal_length']/cam['res_w']*2
            if 'translation' in cam:
                cam['translation'] = cam['translation']/1000 # mm to meters
            
            # Add intrinsic parameters vector
            cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                               cam['center'],
                                               cam['radial_distortion'],
                                               cam['tangential_distortion']))
            
        for subject in subjects_test2:
            self._cameras[subject] = copy.deepcopy(mpi_inf_3dhp_cameras_extrinsic_params['chestHeight'])
            cam = self._cameras[subject] [0]
            cam.update(mpi_inf_3dhp_cameras_intrinsic_params[14])
            for k, v in cam.items():
                if k not in ['id', 'res_w', 'res_h']:
                    cam[k] = np.array(v, dtype='float32')
            
            # Normalize camera frame
            cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
            cam['focal_length'] = cam['focal_length']/cam['res_w']*2
            if 'translation' in cam:
                cam['translation'] = cam['translation']/1000 # mm to meters
            
            # Add intrinsic parameters vector
            cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                               cam['center'],
                                               cam['radial_distortion'],
                                               cam['tangential_distortion']))
        
        # Load serialized dataset
        data = np.load(path, allow_pickle=True)['positions_3d'].item()
        
        self._data = {}
        
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }
                
        if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 32
            self._skeleton.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            
            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8
                
            
    def supports_semi_supervised(self):
        return True
   
