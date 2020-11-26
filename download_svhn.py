import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.externals.joblib import Parallel, delayed


def class_dirname(y):
    return 'class_%s' % y


'''
指定したディレクトリに画像をリサイズして保存（並列処理）
'''


def save_images(images, labels, dirnames):
    r = Parallel(n_jobs=-1, verbose=10)(
        delayed(cv2.imwrite)(os.path.join(dirnames[y], '%s_%s.png' % (class_dirname(y), i)),
                             cv2.cvtColor(x, cv2.COLOR_RGB2BGR)) for i, (x, y) in enumerate(zip(images, labels)))


'''
train, test及びクラスごとのディレクトリの作成
'''


def make_directory(prefix, num_labels, directory):
    # ディレクトリ名の作成
    root_dirname = os.path.join(directory, prefix)
    dirnames = [os.path.join(root_dirname, class_dirname(i)) for i in range(num_labels)]

    # ディレクトリの作成
    os.mkdir(root_dirname)
    for dirname in dirnames:
        os.mkdir(dirname)

    return dirnames


def main():
    # データセットのダウンロード
    tf.enable_eager_execution()
    datasets = tfds.load(name='svhn_cropped', batch_size=-1)
    x_train = tfds.dataset_as_numpy(datasets['train']['image'])
    y_train = tfds.dataset_as_numpy(datasets['train']['label'])
    x_test = tfds.dataset_as_numpy(datasets['test']['image'])
    y_test = tfds.dataset_as_numpy(datasets['test']['label'])

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # ディレクトリの設定
    svhn_directory = './svhn_cropped'
    os.makedirs(svhn_directory)
    num_labels = len(np.unique(y_train))
    dirnames_train = make_directory('train', num_labels, svhn_directory)
    dirnames_test = make_directory('test', num_labels, svhn_directory)

    # 画像の保存
    print('saveing train and test images to %s...\n' % svhn_directory)
    save_images(x_train, y_train, dirnames_train)
    save_images(x_test, y_test, dirnames_test)


if __name__ == '__main__':
    main()
