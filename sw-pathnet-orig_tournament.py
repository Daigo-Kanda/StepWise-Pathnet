import argparse
import copy
import gc
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

import swpathnet_func

# tensorflowの糞メモリ確保回避のおまじない
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)


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
        elif args.model_name == 'resnet50':
            load_pre_model = keras.applications.resnet50.ResNet50
        else:
            sys.stderr('invalid model_name: ', args.model_name)
        pre_model = load_pre_model(input_shape=image_shape,
                                   weights='imagenet',
                                   include_top=True)
    else:
        pre_model = keras.models.load_model(args.trained_model)

    # Step-wise pathnetインスタンスの作成
    pathnet = swpathnet_func.sw_pathnet(pre_model, args.n_comp, args.num_classes, args.transfer_all)

    # 遺伝子型の生成
    li_geopath = [pathnet.gen_geopath() for x in range(args.n_geopath)]
    print(li_geopath)

    # optimizerの指定
    li_opt = [keras.optimizers.Adam() for i in range(args.n_comp)]

    # augumentationの設定
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
    i_comp_log = []
    i_win_log = []

    # 学習
    i_win = 0
    for epoch in range(args.epochs):
        # 遺伝子型の選択（対戦相手）
        li_i_comp = list(np.random.choice(args.n_geopath, size=2, replace=False))

        # それぞれのモデルを1epoch学習
        histories = []
        tmp_weights = []
        for i, i_comp in enumerate(li_i_comp):
            print('%s th, geopath%s is under training' % (epoch, i_comp))

            # モデルの作成，GPU並列
            if args.n_gpu == 1:
                model = pathnet.gene2model(li_geopath[i_comp])
            else:
                with tf.device('/device:CPU:0'):
                    tmp_model = pathnet.gene2model(li_geopath[i_comp])
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
        which_win = np.argmax(tmp_acc)
        i_win = li_i_comp[which_win]

        print('%s th i_comp:%s geopath:%s acc: %s' % (epoch, li_i_comp, li_geopath[i_win], tmp_acc[which_win]))

        # 勝者による上書き ・突然変異
        for i in li_i_comp:
            if i != i_win:
                li_geopath[i] = pathnet.mutate_geopath(np.copy(li_geopath[i_win]))

        # 重みの保存
        for i, i_geopath in enumerate(li_i_comp):
            if i != which_win:
                pathnet.store_weights(li_geopath[i_geopath], tmp_weights[i])
        pathnet.store_weights(li_geopath[i_win], tmp_weights[which_win])

        # ログの格納
        history_log.append(histories[which_win])
        geopath_best_log.append(copy.deepcopy(li_geopath[which_win]))
        geopath_set_log.append(copy.deepcopy(li_geopath))
        i_comp_log.append(li_i_comp)
        i_win_log.append(i_win)

        # ガベコレ
        del tmp_weights
        gc.collect()

    # ログデータの吐き出し
    data = np.array(history_log).reshape((args.epochs, 4))
    df_data = pd.DataFrame(data, columns=['loss', 'acc', 'val_loss', 'val_acc'])
    df_data['geopath_best'] = [str(x) for x in geopath_best_log]
    df_data['geopath_set'] = [str(x) for x in geopath_set_log]
    df_data['i_comp'] = [str(x) for x in i_comp_log]
    df_data['i_win'] = i_win_log
    df_data
    epoch_log_name = 'swpathnet_prev-tournament-%s_%s.csv' % (args.model_name, args.learning_number)
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

    # 遺伝子型の数
    parser.add_argument('--n_geopath', type=int, default=20)

    # 世代数
    parser.add_argument('--epochs', type=int, default=100)

    # 勝負する遺伝子型の数
    parser.add_argument('--n_comp', type=int, default=2)

    # バッチサイズ
    parser.add_argument('--batch_size', type=int, default=128)

    # 1世代で学習するエポック数
    parser.add_argument('--geopath_epochs', type=int, default=1)

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

    # 層の範囲の選択
    #   True: 全て（含むBN）, False: CNN, FCのみ
    parser.add_argument('--transfer_all', action='store_true')

    # （テスト用）学習済みモデルの設定
    parser.add_argument('--trained_model', default=None)

    # 使う学習済みモデル
    parser.add_argument('--model_name', default='vgg16',
                        help='name of pre-trained network. this is disable by giving --trained_model')
    args = parser.parse_args()

    print('----PARSED ARGS----\n%s\n-----------------' % args)

    main(args)