import numpy as np
import os.path as osp
from scipy.io import loadmat
from subprocess import call
from os import makedirs


subject_list = [1, 2, 3, 4, 5, 6, 7, 8]
sequence_list = [1, 2]
camera_list = [0, 1, 2, 4, 5, 6, 7, 8]

makedirs('dataset/mpi_inf_3dhp/images', exist_ok=True)

cnt = 0
for s in subject_list:
    for se in sequence_list:
        for c in camera_list:
            subdir_format = 's_{:02d}_seq_{:02d}_ca_{:02d}'

            subdir = subdir_format.format(s, se, c)
            makedirs(osp.join('dataset/mpi_inf_3dhp/images', subdir), exist_ok=True)

            fileformat = 'dataset/mpi_inf_3dhp/images' + '/' + subdir + '/' + subdir + '_%06d.jpg'

            videopath = 'dataset/mpi_inf_3dhp/S{:01d}/Seq{:01d}/imageSequence/video_{:01d}.avi'.format(s, se, c)
            # print(videoname.split('.')[0])
            subject = 'S' + str(s)

            print(videopath)
            cnt += 1
            call([
                'ffmpeg',
                '-nostats',
                '-i', videopath,
                '-qscale:v', '3',
                fileformat
                    ])




