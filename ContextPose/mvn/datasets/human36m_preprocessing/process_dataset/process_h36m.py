#!/usr/bin/env python3
import argparse
import tarfile
from os import path, makedirs, listdir
from shutil import move
from spacepy import pycdf
import numpy as np
import h5py
from subprocess import call
from tempfile import TemporaryDirectory
from tqdm import tqdm

from metadata import load_h36m_metadata


subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_path", type=str, required=True, help="Path, where raw data file is stored")
    parser.add_argument("--extract_path", type=str, required=True, help="Path, where extracted data file to be stored")
    parser.add_argument("--process_path", type=str, required=True, help="Path, where processed data file to be stored")
    args = parser.parse_args()
    return args



# https://stackoverflow.com/a/6718435
def commonprefix(m):
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


def extract_tgz(tgz_file, dest):
    if path.exists(dest):
        return
    with tarfile.open(tgz_file, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.isreg()]
        member_dirs = [path.dirname(m.name).split(path.sep) for m in members]
        base_path = path.sep.join(commonprefix(member_dirs))
        for m in members:
            m.name = path.relpath(m.name, base_path)
        tar.extractall(dest)


def extract_all(raw_path, extract_path, process_path):
    for subject_id in tqdm(subjects, ascii=True):
        out_dir = path.join(extract_path, subject_id)
        makedirs(out_dir, exist_ok=True)
        extract_tgz(path.join(raw_path, 'Poses_D2_Positions_{}.tgz').format(subject_id),
                    path.join(out_dir, 'Poses_D2_Positions'))
        extract_tgz(path.join(raw_path, 'Poses_D3_Positions_{}.tgz').format(subject_id),
                    path.join(out_dir, 'Poses_D3_Positions')),
        extract_tgz(path.join(raw_path, 'Poses_D3_Positions_mono_{}.tgz').format(subject_id),
                    path.join(out_dir, 'Poses_D3_Positions_mono')),
        extract_tgz(path.join(raw_path, 'Poses_D3_Positions_mono_universal_{}.tgz').format(subject_id),
                    path.join(out_dir, 'Poses_D3_Positions_mono_universal')),
        extract_tgz(path.join(raw_path, 'Videos_{}.tgz').format(subject_id),
                    path.join(out_dir, 'Videos'))
        out_dir = path.join(process_path, subject_id)
        makedirs(out_dir, exist_ok=True)
        extract_tgz(path.join(raw_path, 'Segments_mat_gt_bb_{}.tgz').format(subject_id),
                    path.join(out_dir, 'MySegmentsMat/ground_truth_bb'))


metadata = load_h36m_metadata()

# Subjects to include when preprocessing
included_subjects = {
    'S1': 1,
    'S5': 5,
    'S6': 6,
    'S7': 7,
    'S8': 8,
    'S9': 9,
    'S11': 11,
}

# Sequences with known issues
blacklist = {
    ('S11', '2', '2', '54138969'),  # Video file is corrupted
}

# Rather than include every frame from every video, we can instead wait for the pose to change
# significantly before storing a new example.
def select_frame_indices_to_include(subject, poses_3d_univ):
    # To process every single frame, uncomment the following line:
    # return np.arange(0, len(poses_3d_univ))

    # Take every 64th frame for the protocol #2 test subjects
    # (see the "Compositional Human Pose Regression" paper)
    if subject == 'S9' or subject == 'S11':
        return np.arange(0, len(poses_3d_univ), 64)

    # Take only frames where movement has occurred for the protocol #2 train subjects
    frame_indices = []
    prev_joints3d = None
    threshold = 40 ** 2  # Skip frames until at least one joint has moved by 40mm
    for i, joints3d in enumerate(poses_3d_univ):
        if prev_joints3d is not None:
            max_move = ((joints3d - prev_joints3d) ** 2).sum(axis=-1).max()
            if max_move < threshold:
                continue
        prev_joints3d = joints3d
        frame_indices.append(i)
    return np.array(frame_indices)


def infer_camera_intrinsics(points2d, points3d):
    """Infer camera instrinsics from 2D<->3D point correspondences."""
    pose2d = points2d.reshape(-1, 2)
    pose3d = points3d.reshape(-1, 3)
    x3d = np.stack([pose3d[:, 0], pose3d[:, 2]], axis=-1)
    x2d = (pose2d[:, 0] * pose3d[:, 2])
    alpha_x, x_0 = list(np.linalg.lstsq(x3d, x2d, rcond=-1)[0].flatten())
    y3d = np.stack([pose3d[:, 1], pose3d[:, 2]], axis=-1)
    y2d = (pose2d[:, 1] * pose3d[:, 2])
    alpha_y, y_0 = list(np.linalg.lstsq(y3d, y2d, rcond=-1)[0].flatten())
    return np.array([alpha_x, x_0, alpha_y, y_0])


def process_view(out_dir, subject, action, subaction, camera, extract_path):
    subj_dir = path.join(extract_path, subject)

    base_filename = metadata.get_base_filename(subject, action, subaction, camera)

    # Load joint position annotations
    with pycdf.CDF(path.join(subj_dir, 'Poses_D2_Positions', base_filename + '.cdf')) as cdf:
        poses_2d = np.array(cdf['Pose'])
        poses_2d = poses_2d.reshape(poses_2d.shape[1], 32, 2)
    with pycdf.CDF(path.join(subj_dir, 'Poses_D3_Positions_mono_universal', base_filename + '.cdf')) as cdf:
        poses_3d_univ = np.array(cdf['Pose'])
        poses_3d_univ = poses_3d_univ.reshape(poses_3d_univ.shape[1], 32, 3)
    with pycdf.CDF(path.join(subj_dir, 'Poses_D3_Positions_mono', base_filename + '.cdf')) as cdf:
        poses_3d = np.array(cdf['Pose'])
        poses_3d = poses_3d.reshape(poses_3d.shape[1], 32, 3)

    # Infer camera intrinsics
    camera_int = infer_camera_intrinsics(poses_2d, poses_3d)
    camera_int_univ = infer_camera_intrinsics(poses_2d, poses_3d_univ)

    frame_indices = select_frame_indices_to_include(subject, poses_3d_univ)
    frames = frame_indices + 1
    video_file = path.join(subj_dir, 'Videos', base_filename + '.mp4')
    frames_dir = path.join(out_dir, 'imageSequence', camera)
    makedirs(frames_dir, exist_ok=True)

    # Check to see whether the frame images have already been extracted previously
    existing_files = {f for f in listdir(frames_dir)}
    frames_are_extracted = True
    for i in frames:
        filename = 'img_%06d.jpg' % i
        if filename not in existing_files:
            frames_are_extracted = False
            break

    if not frames_are_extracted:
        with TemporaryDirectory() as tmp_dir:
            # Use ffmpeg to extract frames into a temporary directory
            call([
                'ffmpeg',
                '-nostats', '-loglevel', '0',
                '-i', video_file,
                '-qscale:v', '3',
                path.join(tmp_dir, 'img_%06d.jpg')
            ])

            # Move included frame images into the output directory
            for i in frames:
                filename = 'img_%06d.jpg' % i
                move(
                    path.join(tmp_dir, filename),
                    path.join(frames_dir, filename)
                )

    return {
        'pose/2d': poses_2d[frame_indices],
        'pose/3d-univ': poses_3d_univ[frame_indices],
        'pose/3d': poses_3d[frame_indices],
        'intrinsics/' + camera: camera_int,
        'intrinsics-univ/' + camera: camera_int_univ,
        'frame': frames,
        'camera': np.full(frames.shape, int(camera)),
        'subject': np.full(frames.shape, int(included_subjects[subject])),
        'action': np.full(frames.shape, int(action)),
        'subaction': np.full(frames.shape, int(subaction)),
    }


def process_subaction(subject, action, subaction, extract_path, process_path):
    datasets = {}

    out_dir = path.join(process_path, subject, metadata.action_names[action] + '-' + subaction)
    makedirs(out_dir, exist_ok=True)

    for camera in tqdm(metadata.camera_ids, ascii=True, leave=False):
        if (subject, action, subaction, camera) in blacklist:
            continue

        try:
            annots = process_view(out_dir, subject, action, subaction, camera, extract_path)
        except:
            tqdm.write('!!! Error processing sequence, skipping: ' + \
                       repr((subject, action, subaction, camera)))
            tqdm.write(traceback.format_exc())
            continue

        for k, v in annots.items():
            if k in datasets:
                datasets[k].append(v)
            else:
                datasets[k] = [v]

    if len(datasets) == 0:
        return

    datasets = {k: np.concatenate(v) for k, v in datasets.items()}

    with h5py.File(path.join(out_dir, 'annot.h5'), 'w') as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=data)


def process_all(extract_path, process_path):
    sequence_mappings = metadata.sequence_mappings

    subactions = []

    for subject in included_subjects.keys():
        subactions += [
            (subject, action, subaction)
            for action, subaction in sequence_mappings[subject].keys()
            if int(action) > 1  # Exclude '_ALL'
        ]

    for subject, action, subaction in tqdm(subactions, ascii=True, leave=False):
        process_subaction(subject, action, subaction, extract_path, process_path)


if __name__ == '__main__':
    args = parse_args()
    raw_path = args.raw_path
    extract_path = args.extract_path
    process_path = args.process_path
    print('==> start extracting ...')
    print("extracting data from {} to {}".format(raw_path, extract_path))
    print("extracting SegmentsMat from {} to {}".format(raw_path, process_path))
    extract_all(raw_path, extract_path, process_path)

    print('==> start processing ...')
    print("processing extracted data now")
    process_all(extract_path, process_path)
