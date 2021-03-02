import math
import os
import os.path

import numpy as np
import scipy.io as sio
import tensorflow.keras.preprocessing.image as pre
from PIL import Image
from tensorflow.keras.utils import Sequence

'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.
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

MEAN_PATH = '../'


def loadMetadata(filename, silent=False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True,
                               struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


class ITrackerData(Sequence):
    def __init__(self, dataPath, split='train', imSize=(224, 224), gridSize=(25, 25), batch_size=32):

        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize

        print('Loading iTracker dataset...')
        metaFile = os.path.join(dataPath, '../metadata.mat')
        # metaFile = 'metadata.mat'
        if metaFile is None or not os.path.isfile(metaFile):
            raise RuntimeError(
                'There is no such file %s! Provide a valid dataset path.' % metaFile)
        self.metadata = loadMetadata(metaFile)
        if self.metadata is None:
            raise RuntimeError(
                'Could not read metadata file %s! Provide a valid dataset path.' % metaFile)

        self.faceMean = loadMetadata(os.path.join(
            MEAN_PATH, '../mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(
            MEAN_PATH, '../mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(
            MEAN_PATH, '../mean_right_224.mat'))['image_mean']

        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        else:
            mask = self.metadata['labelTrain']

        self.indices = np.argwhere(mask)[:, 0]
        self.batch_size = batch_size
        print('Loaded iTracker dataset split "%s" with %d records...' %
              (split, len(self.indices)))

    def transform(self, input, mean):

        x = input.resize(self.imSize)
        x = pre.img_to_array(x)
        x = (x / 255) - (mean / 255)

        return x

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            # im = Image.new("RGB", self.imSize, "white")

        return im

    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen, ], np.float32)

        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(
            indsX >= params[0], indsX < params[0] + params[2])
        condY = np.logical_and(
            indsY >= params[1], indsY < params[1] + params[3])
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):

        # create batch structures
        left_eye_batch = np.zeros(
            shape=(self.batch_size, self.imSize[0], self.imSize[1], 3), dtype=np.float32)
        right_eye_batch = np.zeros(
            shape=(self.batch_size, self.imSize[0], self.imSize[1], 3), dtype=np.float32)
        face_batch = np.zeros(
            shape=(self.batch_size, self.imSize[0], self.imSize[1], 3), dtype=np.float32)
        face_grid_batch = np.zeros(
            shape=(self.batch_size, 25 * 25, 1), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, 2), dtype=np.float32)

        for i in range(self.batch_size):

            if (self.batch_size * index + i) >= self.indices.shape[0]:
                dummy = np.abs(
                    self.indices.shape[0] - (self.batch_size * index + i))
            else:
                dummy = (self.batch_size * index + i)

            metaIndex = self.indices[dummy]

            imFacePath = os.path.join(self.dataPath, '%05d/appleFace/%05d.jpg' % (
                self.metadata['labelRecNum'][metaIndex], self.metadata['frameIndex'][metaIndex]))
            imEyeLPath = os.path.join(self.dataPath, '%05d/appleLeftEye/%05d.jpg' % (
                self.metadata['labelRecNum'][metaIndex], self.metadata['frameIndex'][metaIndex]))
            imEyeRPath = os.path.join(self.dataPath, '%05d/appleRightEye/%05d.jpg' % (
                self.metadata['labelRecNum'][metaIndex], self.metadata['frameIndex'][metaIndex]))

            imFace = self.loadImage(imFacePath)
            imEyeL = self.loadImage(imEyeLPath)
            imEyeR = self.loadImage(imEyeRPath)

            imFace = self.transform(imFace, self.faceMean)
            imEyeL = self.transform(imEyeL, self.eyeLeftMean)
            imEyeR = self.transform(imEyeR, self.eyeRightMean)

            gaze = np.array([self.metadata['labelDotXCam'][metaIndex],
                             self.metadata['labelDotYCam'][metaIndex]], np.float32)

            faceGrid = self.makeGrid(
                self.metadata['labelFaceGrid'][metaIndex, :])

            right_eye_batch[i] = imEyeR
            left_eye_batch[i] = imEyeL
            face_batch[i] = imFace
            face_grid_batch[i] = faceGrid[:, np.newaxis]
            y_batch[i] = gaze

        return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch

    # エポック数の返却
    def __len__(self):
        # a = math.ceil(len(self.indices) / self.batch_size)
        return math.ceil(len(self.indices) / self.batch_size)
