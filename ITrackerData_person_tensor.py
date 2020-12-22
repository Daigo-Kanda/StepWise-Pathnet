import gc
import os
import os.path
import random

import numpy as np
import scipy.io as sio
import tensorflow as tf

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


def preprocess_image(image, size, mean):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [size, size])
    image = (image / 255.0) - (mean / 255.0)

    return image


def load_and_preprocess_image(path, mean, size):
    image = tf.io.read_file(path)
    return preprocess_image(image, mean, size)


def load_image(path):
    image = tf.io.read_file(path)
    return image


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


def makeGrid(all_grid, gridSize):
    gridLen = gridSize[0] * gridSize[1]
    grids = np.zeros([len(all_grid), gridLen], np.int8)

    indsY = np.tile(np.array([i // gridSize[0] for i in range(gridLen)], np.int8), (len(all_grid), 1))
    indsX = np.tile(np.array([i % gridSize[0] for i in range(gridLen)], np.int8), (len(all_grid), 1))

    condX = np.logical_and(
        indsX >= all_grid[:, 0, np.newaxis], indsX < all_grid[:, 0, np.newaxis] + all_grid[:, 2, np.newaxis])
    condY = np.logical_and(
        indsY >= all_grid[:, 1, np.newaxis], indsY < all_grid[:, 1, np.newaxis] + all_grid[:, 3, np.newaxis])
    cond = np.logical_and(condX, condY)

    grids[cond] = 1

    del condX, condY, cond
    gc.collect()

    return grids


def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def shuffle(obj):
    set_seed()
    np.random.shuffle(obj)


def getData(batch_size, memory_size, dataset_path):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    gridSize = (25, 25)

    dataPath = dataset_path
    metaFile = os.path.join(dataPath, 'metadata_person.mat')
    metadata = loadMetadata(metaFile)

    MEAN_PATH = './'
    faceMean = loadMetadata(os.path.join(
        MEAN_PATH, 'mean_face_224.mat'))['image_mean']
    eyeLeftMean = loadMetadata(os.path.join(
        MEAN_PATH, 'mean_left_224.mat'))['image_mean']
    eyeRightMean = loadMetadata(os.path.join(
        MEAN_PATH, 'mean_right_224.mat'))['image_mean']

    recordNum = metadata['labelRecNum']
    frameIndex = metadata['frameIndex']

    # string_list = ["Face", "RightEye", "LeftEye"]
    # for i in range(3):

    all_face_list = [os.path.join(dataPath, 'appleFace/%05d.jpg' % frame)
                     for frame in frameIndex]
    all_right_list = [os.path.join(dataPath, 'appleRightEye/%05d.jpg' % frame)
                      for frame in frameIndex]
    all_left_list = [os.path.join(dataPath, 'appleLeftEye/%05d.jpg' % frame)
                     for frame in frameIndex]
    all_grid_list = makeGrid(metadata['labelFaceGrid'], gridSize)
    all_gaze_list = [np.array([x, y], np.float32) for x, y in
                     zip(metadata['labelDotXCam'], metadata['labelDotYCam'])]

    shuffle(all_face_list)
    shuffle(all_right_list)
    shuffle(all_left_list)
    shuffle(all_grid_list)
    shuffle(all_gaze_list)

    DATASET_SIZE = len(all_face_list)

    face_path_ds = tf.data.Dataset.from_tensor_slices(all_face_list)
    right_path_ds = tf.data.Dataset.from_tensor_slices(all_right_list)
    left_path_ds = tf.data.Dataset.from_tensor_slices(all_left_list)

    grid_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_grid_list, tf.int8))
    gaze_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_gaze_list, tf.float32))

    del all_face_list, all_right_list, all_left_list, all_grid_list, all_gaze_list
    gc.collect()

    size = 224

    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    train_ds = face_path_ds.take(train_size)
    remaining = face_path_ds.skip(train_size)
    valid_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    face_path_ds_list = [train_ds, valid_ds, test_ds]
    print(id(face_path_ds_list[0]))

    train_ds = right_path_ds.take(train_size)
    remaining = right_path_ds.skip(train_size)
    valid_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    right_path_ds_list = [train_ds, valid_ds, test_ds]
    print(id(right_path_ds_list[0]))

    train_ds = left_path_ds.take(train_size)
    remaining = left_path_ds.skip(train_size)
    valid_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    left_path_ds_list = [train_ds, valid_ds, test_ds]
    print(id(left_path_ds_list[0]))

    train_ds = grid_ds.take(train_size)
    remaining = grid_ds.skip(train_size)
    valid_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    grid_ds_list = [train_ds, valid_ds, test_ds]

    train_ds = gaze_ds.take(train_size)
    remaining = gaze_ds.skip(train_size)
    valid_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    gaze_ds_list = [train_ds, valid_ds, test_ds]

    data_list = []
    for i in range(3):

        if memory_size > 100:
            face_img_ds = face_path_ds_list[i].map(lambda x: load_image(x),
                                                   num_parallel_calls=AUTOTUNE)
            face_img_ds = face_img_ds.map(
                lambda x: preprocess_image(x, size, faceMean),
                num_parallel_calls=AUTOTUNE)
            right_img_ds = right_path_ds_list[i].map(lambda x: load_image(x),
                                                     num_parallel_calls=AUTOTUNE)
            right_img_ds = right_img_ds.map(
                lambda x: preprocess_image(x, size, eyeRightMean),
                num_parallel_calls=AUTOTUNE)
            left_img_ds = left_path_ds_list[i].map(lambda x: load_image(x),
                                                   num_parallel_calls=AUTOTUNE)
            left_img_ds = left_img_ds.map(
                lambda x: preprocess_image(x, size, eyeLeftMean),
                num_parallel_calls=AUTOTUNE)

        else:
            face_img_ds = face_path_ds_list[i].map(lambda x: load_and_preprocess_image(x, size, faceMean),
                                                   num_parallel_calls=AUTOTUNE)
            right_img_ds = right_path_ds_list[i].map(lambda x: load_and_preprocess_image(x, size, eyeRightMean),
                                                     num_parallel_calls=AUTOTUNE)
            left_img_ds = left_path_ds_list[i].map(lambda x: load_and_preprocess_image(x, size, eyeLeftMean),
                                                   num_parallel_calls=AUTOTUNE)

        data = tf.data.Dataset.zip(((right_img_ds, left_img_ds, face_img_ds, grid_ds_list[i]), gaze_ds_list[i]))

        if batch_size != 0:
            ds = data.batch(batch_size)
        else:
            ds = data

        ds = ds.prefetch(buffer_size=AUTOTUNE)

        data_list.append(ds)

    return data_list
