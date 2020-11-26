import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import scipy.io as sio

'''
Prepares the GazeCapture dataset for use with the pytorch code. Crops images, compiles JSONs into metadata.mat

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

parser = argparse.ArgumentParser(description='iTracker-pytorch-PrepareDataset.')
parser.add_argument('--dataset_path', help="Path to extracted files. It should have folders called '%%05d' in it.")
parser.add_argument('--output_path', default=None,
                    help="Where to write the output. Can be the same as dataset_path if you wish (=default).")
args = parser.parse_args()


def main():
    if args.output_path is None:
        args.output_path = args.dataset_path

    if args.dataset_path is None or not os.path.isdir(args.dataset_path):
        raise RuntimeError('No such dataset folder %s!' % args.dataset_path)

    # preparePath(args.output_path)

    # list recordings
    recordings = os.listdir(args.dataset_path)
    recordings = np.array(recordings, np.object)
    recordings = recordings[[os.path.isdir(os.path.join(args.dataset_path, r)) for r in recordings]]
    recordings.sort()

    # loop for every recordings
    for i, recording in enumerate(recordings):

        # Output structure
        meta = {
            'labelRecNum': [],
            'frameIndex': [],
            'labelDotXCam': [],
            'labelDotYCam': [],
            'labelFaceGrid': [],
        }

        print('[%d/%d] Processing recording %s (%.2f%%)' % (i, len(recordings), recording, i / len(recordings) * 100))
        recDir = os.path.join(args.dataset_path, recording)
        recDirOut = os.path.join(args.output_path, recording)

        # Read JSONs
        appleFace = readJson(os.path.join(recDir, 'appleFace.json'))
        if appleFace is None:
            continue
        appleLeftEye = readJson(os.path.join(recDir, 'appleLeftEye.json'))
        if appleLeftEye is None:
            continue
        appleRightEye = readJson(os.path.join(recDir, 'appleRightEye.json'))
        if appleRightEye is None:
            continue
        dotInfo = readJson(os.path.join(recDir, 'dotInfo.json'))
        if dotInfo is None:
            continue
        faceGrid = readJson(os.path.join(recDir, 'faceGrid.json'))
        if faceGrid is None:
            continue
        frames = readJson(os.path.join(recDir, 'frames.json'))
        if frames is None:
            continue
        # info = readJson(os.path.join(recDir, 'info.json'))
        # if info is None:
        #     continue
        # screen = readJson(os.path.join(recDir, 'screen.json'))
        # if screen is None:
        #     continue

        # facePath = preparePath(os.path.join(recDirOut, 'appleFace'))
        # leftEyePath = preparePath(os.path.join(recDirOut, 'appleLeftEye'))
        # rightEyePath = preparePath(os.path.join(recDirOut, 'appleRightEye'))

        # Preprocess
        allValid = np.logical_and(np.logical_and(appleFace['IsValid'], appleLeftEye['IsValid']),
                                  np.logical_and(appleRightEye['IsValid'], faceGrid['IsValid']))
        if not np.any(allValid):
            continue

        frames = np.array([int(re.match('(\d{5})\.jpg$', x).group(1)) for x in frames])

        bboxFromJson = lambda data: np.stack((data['X'], data['Y'], data['W'], data['H']), axis=1).astype(int)
        faceBbox = bboxFromJson(appleFace) + [-1, -1, 1, 1]  # for compatibility with matlab code
        leftEyeBbox = bboxFromJson(appleLeftEye) + [0, -1, 0, 0]
        rightEyeBbox = bboxFromJson(appleRightEye) + [0, -1, 0, 0]
        leftEyeBbox[:, :2] += faceBbox[:, :2]  # relative to face
        rightEyeBbox[:, :2] += faceBbox[:, :2]
        faceGridBbox = bboxFromJson(faceGrid)

        # make eye and face images
        for j, frame in enumerate(frames):
            # Can we use it?
            if not allValid[j]:
                continue

            # Collect metadata
            meta['labelRecNum'] += [int(recording)]
            meta['frameIndex'] += [frame]
            meta['labelDotXCam'] += [dotInfo['XCam'][j]]
            meta['labelDotYCam'] += [dotInfo['YCam'][j]]
            meta['labelFaceGrid'] += [faceGridBbox[j, :]]

        # Integrate
        meta['labelRecNum'] = np.stack(meta['labelRecNum'], axis=0).astype(np.int16)
        meta['frameIndex'] = np.stack(meta['frameIndex'], axis=0).astype(np.int32)
        meta['labelDotXCam'] = np.stack(meta['labelDotXCam'], axis=0)
        meta['labelDotYCam'] = np.stack(meta['labelDotYCam'], axis=0)
        meta['labelFaceGrid'] = np.stack(meta['labelFaceGrid'], axis=0).astype(np.uint8)

        # percentage of data
        num_all = len(meta['labelRecNum'])
        num_train = int(0.8 * num_all)
        num_test = int(0.1 * num_all)
        num_validation = int(0.1 * num_all)

        # indices of randomized data
        id_all = np.random.choice(num_all, num_all, replace=False)
        id_train = id_all[0:num_train]
        id_test = id_all[num_train:num_train + num_test]
        id_validation = id_all[num_train + num_test:]

        meta['labelTrain'] = np.zeros((len(meta['labelRecNum'], )), np.bool)
        meta['labelTrain'][id_train] = 1
        meta['labelTest'] = np.zeros((len(meta['labelRecNum'], )), np.bool)
        meta['labelTest'][id_test] = 1
        meta['labelVal'] = np.zeros((len(meta['labelRecNum'], )), np.bool)
        meta['labelVal'][id_validation] = 1

        # b = np.sum(meta['labelTrain'] == 1) + np.sum(meta['labelTest'] == 1) + np.sum(meta['labelVal'] == 1)
        # a = np.logical_and(np.logical_and(meta['labelTrain'], meta['labelVal']),meta['labelTest'])

        # Write out metadata
        metaFile = os.path.join(recDirOut, 'metadata_person.mat')
        print('Writing out the metadata.mat to %s...' % metaFile)
        sio.savemat(metaFile, meta)

    # import pdb; pdb.set_trace()
    input("Press Enter to continue...")


def readJson(filename):
    if not os.path.isfile(filename):
        logError('Warning: No such file %s!' % filename)
        return None

    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = None

    if data is None:
        logError('Warning: Could not read file %s!' % filename)
        return None

    return data


def preparePath(path, clear=False):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path


def logError(msg, critical=False):
    print(msg)
    if critical:
        sys.exit(1)


def cropImage(img, bbox):
    bbox = np.array(bbox, int)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

    aDst = aSrc - bbox[:2]
    bDst = aDst + (bSrc - aSrc)

    res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)
    res[aDst[1]:bDst[1], aDst[0]:bDst[0], :] = img[aSrc[1]:bSrc[1], aSrc[0]:bSrc[0], :]

    return res


if __name__ == "__main__":
    main()
    print('DONE')
