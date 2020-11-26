import argparse
import copy
import gc
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

import swpathnet_func

# tensorflowの糞メモリ確保回避のおまじない
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class AntColony:
    def __init__(self, n_layer,
                 evapolate_rate=1e-3,
                 initialize_method='uniform',
                 offset=0.1):
        # レイヤ数の設定
        self.n_layer = n_layer

        # 蒸発率の設定
        self.evapolate_rate = evapolate_rate

        # フェロモンの初期化
        #   フェロモンは学習済みを選ぶ確率
        coef_offset = 1.0 - offset * 2
        if initialize_method == 'uniform':
            self.pheromone = np.array([0.5] * n_layer)

        elif initialize_method == 'linear':
            self.pheromone = np.array([(1.0 - x / (n_layer - 1)) * coef_offset + offset for x in range(n_layer)])

        elif initialize_method == 'exp':
            self.pheromone = np.array([np.exp(-x) * coef_offset + offset for x in range(n_layer)])

        else:
            sys.stderr('bad initialize_mothod %s' % initialize_method)
        print('initialized pheromone: %s' % self.pheromone)

    # フェロモンでgeopathを生成
    def gen_geopath(self, is_flatten=False):
        # geopathの生成
        geopath = np.array([
            np.random.choice(
                [0, 1],
                p=[self.pheromone[i], 1.0 - self.pheromone[i]])
            for i in range(self.n_layer)])

        # flatten処理
        if is_flatten:
            i_switch = np.where(geopath == 1)[0]
            if i_switch.size != 0:
                geopath[i_switch[0]:] = 1

        # 最後尾の識別層の追加
        geopath = np.append(geopath, 1)

        return geopath

    # フェロモンの更新
    def update_pheromone(self, li_geopath, li_acc):
        update_rate = 1.0 / len(li_geopath)

        # 蒸発(0.5に近づかせる)
        diff = self.pheromone - 0.5
        self.pheromone -= diff * update_rate

        # 各解候補による付与
        for geopath, acc in zip(li_geopath, li_acc):
            # 学習済みレイヤのフェロモン付与　
            diff_pretrained = 1.0 - self.pheromone
            mask_pretrained = (1 - np.array(geopath[:self.n_layer]))
            self.pheromone += mask_pretrained * diff_pretrained * acc * update_rate

            # 学習可能レイヤーのフェロモン付与
            diff_adjustable = self.pheromone
            mask_adjustable = np.array(geopath[:self.n_layer])
            self.pheromone -= mask_adjustable * diff_adjustable * acc * update_rate

        # 0-1の間になるようクリップ
        np.clip(self.pheromone, 0, 1, out=self.pheromone)

        print(self.pheromone)


def main(args):
    # 画像サイズの設定
    image_shape = (args.image_size, args.image_size, 3)

    # 学習済みモデルの読み込み
    if args.trained_model is None:
        if args.model_name == 'vgg16':
            load_pre_model = keras.applications.vgg16.VGG16
        elif args.model_name == 'xception':
            load_pre_model = keras.applications.xception.Xception
        elif args.model_name == 'inceptionv3':
            load_pre_model = keras.applications.inception_v3.InceptionV3
        elif args.model_name == 'inceptionresnetv2':
            load_pre_model = keras.applications.inception_resnet_v2.InceptionResNetV2
        elif args.model_name == 'densenet':
            load_pre_model = keras.applications.densenet.DenseNet121
        else:
            sys.stderr('invalid model_name: ', args.model_name)
        pre_model = load_pre_model(input_shape=image_shape,
                                   weights='imagenet',
                                   include_top=True)
    else:
        pre_model = keras.models.load_model(args.trained_model)

    # Step-wise pathnetインスタンスの作成
    pathnet = swpathnet_func.sw_pathnet(pre_model, args.n_comp, args.num_classes)

    # (ACO)ACOインスタンスの生成，フェロモンの初期化
    aco = AntColony(pathnet.len_geopath - 1,
                    evapolate_rate=1 / args.epochs,
                    initialize_method=args.initialize_method,
                    offset=args.pheromone_offset)

    # optimizerの指定
    li_opt = [keras.optimizers.Adam() for i in range(args.n_comp)]

    #   https://github.com/geifmany/cifar-vgg/blob/master/cifar100vgg.pyより拝借
    if args.use_augument:
        train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    else:
        train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    # generatorの設定
    train_generator = train_datagen.flow_from_directory(
        os.path.join(args.dataset_dir, 'train'),
        target_size=image_shape[:2],
        batch_size=args.batch_size,
        class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(
        os.path.join(args.dataset_dir, 'test'),
        target_size=image_shape[:2],
        batch_size=args.batch_size,
        class_mode='categorical')

    # ログ用
    history_log = []
    geopath_best_log = []
    geopath_set_log = []
    i_win_log = []
    pheromone_log = []

    # 学習
    best_geopath_prev = np.array([1] * pathnet.len_geopath)
    for epoch in range(args.epochs):
        # (ACO)geopath生成
        li_geopath = [best_geopath_prev]
        li_geopath += [aco.gen_geopath(is_flatten=args.flatten) for i in range(args.n_comp - 1)]
        print(li_geopath)

        # それぞれのモデルを1epoch学習
        histories = []
        tmp_weights = []
        for i, geopath in enumerate(li_geopath):
            print('%s th, geopath%s is under training' % (epoch, i))
            # モデルの作成，GPU並列
            if args.n_gpu == 1:
                model = pathnet.gene2model(geopath)
            else:
                with tf.device('/device:CPU:0'):
                    tmp_model = pathnet.gene2model(geopath)
                model = keras.utils.multi_gpu_model(tmp_model,
                                                    gpus=args.n_gpu)

            # コンパイル
            model.compile(loss='categorical_crossentropy',
                          optimizer=li_opt[i],
                          metrics=['accuracy'])

            # Fit the model on the batches generated by datagen.flow().
            history = model.fit_generator(
                train_generator,
                steps_per_epoch=args.num_images_train // args.batch_size,
                epochs=args.geopath_epochs,
                validation_data=test_generator,
                validation_steps=args.num_images_test // args.batch_size,
                use_multiprocessing=args.use_multiprocessing,
                workers=args.n_thread,
                verbose=2)

            # 学習結果の格納
            history = np.array([
                [history.history['loss'][-1], history.history['acc'][-1], history.history['val_loss'][-1],
                 history.history['val_acc'][-1]]
            ])
            histories.append(copy.deepcopy(history))

            # 重みの格納
            if args.n_gpu == 1:
                tmp_weights.append(pathnet.extract_weights(model))
            else:
                tmp_weights.append(pathnet.extract_weights(tmp_model))

        # 結果の集計
        tmp_acc = [history[0, 1] for history in histories]
        i_win = np.argmax(tmp_acc)

        # (ACO)フェロモンの更新
        aco.update_pheromone(li_geopath, tmp_acc)

        print('%s th geopath:%s acc: %s pheromone: %s' % (epoch, li_geopath[i_win], tmp_acc[i_win], aco.pheromone))

        # 重みの保存
        for i, geopath in enumerate(li_geopath):
            if i != i_win:
                pathnet.store_weights(geopath, tmp_weights[i])
        pathnet.store_weights(geopath, tmp_weights[i_win])

        # ログの格納
        history_log.append(histories[i_win])
        geopath_best_log.append(copy.deepcopy(li_geopath[i_win]))
        geopath_set_log.append(copy.deepcopy(li_geopath))
        i_win_log.append(i_win)
        pheromone_log.append(copy.deepcopy(aco.pheromone))

        # ガベコレ
        del tmp_weights
        gc.collect()

    # ログデータの吐き出し
    data = np.array(history_log).reshape((args.epochs, 4))
    df_data = pd.DataFrame(data, columns=['loss', 'acc', 'val_loss', 'val_acc'])
    df_data['geopath_best'] = [str(x) for x in geopath_best_log]
    df_data['geopath_set'] = [str(x) for x in geopath_set_log]
    df_data['i_win'] = i_win_log
    df_data['pheromone'] = [str(list(x)) for x in pheromone_log]
    df_data
    epoch_log_name = 'swpathnet_aco-%s.csv' % args.learning_number
    df_data.to_csv(os.path.join(args.save_dir, epoch_log_name), header=True)


# 引数の読み込み
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # N回回す実験用のアレ
    parser.add_argument('learning_number', help='for sequential experiment')

    # データセットのディレクトリ
    parser.add_argument('dataset_dir')

    # ログの保存先ディレクトリ
    parser.add_argument('save_dir')

    # クラス数
    parser.add_argument('num_classes', type=int)

    # 学習データの画像数
    #   CIFAR10: 50000, CIFAR100: 50000
    parser.add_argument('num_images_train', type=int, help='CIFAR10: 50000, CIFAR100: 50000')

    # テストデータの画像数
    #   CIFAR10: 10000, CIFAR100: 10000
    parser.add_argument('num_images_test', type=int, help='CIFAR10: 10000, CIFAR100: 10000')

    # 画像サイズ
    parser.add_argument('--image_size', type=int, default=224)

    # 世代数
    parser.add_argument('--epochs', type=int, default=100)

    # 勝負する遺伝子型の数
    parser.add_argument('--n_comp', type=int, default=2)

    # バッチサイズ
    parser.add_argument('--batch_size', type=int, default=32)

    # 1世代で学習するエポック数
    parser.add_argument('--geopath_epochs', type=int, default=1)

    # フェロモンの初期化方法
    parser.add_argument('--initialize_method', type=str, default='uniform')

    # フェロモン初期化時のoffset
    parser.add_argument('--pheromone_offset', type=float, default=0)

    # flatな構造のみを最適化
    parser.add_argument('--flatten', action='store_true')

    # GPU並列
    parser.add_argument('--n_gpu', type=int, default=1)

    # CPU並列
    parser.add_argument('--n_thread', type=int, default=1)

    # CPU(スレッド)並列
    #   fit_generatorでスレッド並列するとデッドロックする臭い？
    #   https://github.com/keras-team/keras/issues/10340
    parser.add_argument('--use_multiprocessing', action='store_true')

    # 水増しの有無
    parser.add_argument('--use_augument', action='store_true')

    # （テスト用）学習済みモデルの設定
    parser.add_argument('--trained_model', default=None)

    # 使う学習済みモデル
    parser.add_argument('--model_name', default='vgg16',
                        help='name of pre-trained network. this is disable by giving --trained_model')
    args = parser.parse_args()

    args = parser.parse_args()
    print('----PARSED ARGS----\n%s\n-----------------' % args)

    main(args)
