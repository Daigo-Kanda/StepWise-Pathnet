import os

import cv2
import numpy as np
from sklearn.externals.joblib import Parallel, delayed


def class_dirname(y):
    return 'class_%s' % y


def save_images(images, labels, dirnames):
    r = Parallel(n_jobs=-1, verbose=10)(
        delayed(cv2.imwrite)(os.path.join(dirnames[y], '%s_%s.png' % (class_dirname(y), i)),
                             cv2.cvtColor(x, cv2.COLOR_RGB2BGR)) for i, (x, y) in enumerate(zip(images, labels)))
    # bar = progressbar.ProgressBar(max_value=len(images))
    # for i, (x, y) in enumerate(zip(images, labels)):
    #    cv2.imwrite(os.path.join(dirnames[y], 
    #                             '%s_%s.png' % (class_dirname(y), i)), 
    #                cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
    #    bar.update(i)


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
    #########
    # cifar10
    #########
    # データセットのダウンロード
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # ディレクトリの設定
    cifar10_directory = './cifar/cifar10'
    os.makedirs(cifar10_directory)
    num_labels = len(np.unique(y_train))
    dirnames_train = make_directory('train', num_labels, cifar10_directory)
    dirnames_test = make_directory('test', num_labels, cifar10_directory)

    # 画像の保存
    print('saveing train and test images to %s...\n' % cifar10_directory)
    save_images(x_train, y_train, dirnames_train)
    save_images(x_test, y_test, dirnames_test)

    ##########
    # cifar100
    ##########
    # データセットのダウンロード
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # ディレクトリの設定
    cifar100_directory = './cifar/cifar100'
    os.makedirs(cifar100_directory)
    num_labels = len(np.unique(y_train))
    dirnames_train = make_directory('train', num_labels, cifar100_directory)
    dirnames_test = make_directory('test', num_labels, cifar100_directory)

    # 画像の保存
    print('saveing train and test images to %s...\n' % cifar100_directory)
    save_images(x_train, y_train, dirnames_train)
    save_images(x_test, y_test, dirnames_test)


if __name__ == '__main__':
    main()
