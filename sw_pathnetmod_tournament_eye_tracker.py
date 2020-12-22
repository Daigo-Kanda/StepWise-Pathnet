import argparse
import copy
import datetime
import gc
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import ITrackerData_person_tensor as data_gen
import swpathnet_func_eyetracker


def main(args):
    # 画像サイズの設定
    image_shape = (args.image_size, args.image_size, 3)

    pre_model = keras.models.load_model(args.trained_model)

    # Step-wise pathnetインスタンスの作成
    pathnet = swpathnet_func_eyetracker.sw_pathnet(pre_model, args.n_comp, args.transfer_all,
                                                   is_reuse_initweight=args.finetune)

    # list of gene
    li_geopath = [pathnet.gen_geopath(bias_pretrained=1.0 - x / args.n_geopath) for x in range(args.n_geopath)]
    # li_geopath = [pathnet.gen_geopath(bias_pretrained=0.5) for x in range(args.n_geopath)]

    # for eye tracker
    pathnet.adjustGeopath(li_geopath)

    print(li_geopath)

    # optimizerの指定
    li_opt = [keras.optimizers.Adam() for i in range(args.n_comp)]

    # augmentationの設定
    #   https://github.com/geifmany/cifar-vgg/blob/master/cifar100vgg.pyより拝借
    if not args.dont_augment:
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

    # generator setting
    # train_generator = data_gen.ITrackerData(args.dataset_dir, 'train', (args.image_size, args.image_size), (25, 25),
    #                                         args.batch_size)
    # validation_generator = data_gen.ITrackerData(args.dataset_dir, 'val', (args.image_size, args.image_size), (25, 25),
    #                                        args.batch_size)

    data = data_gen.getData(batch_size=args.batch_size, memory_size=120, dataset_path=args.dataset_dir)
    train_generator = data[0]
    validation_generator = data[1]

    # ログ用
    history_log = []
    geopath_best_log = []
    geopath_set_log = []
    i_comp_log = []
    i_win_log = []

    # 学習
    i_win = 0

    # save best val_loss model
    best_val_loss = 1000
    now = datetime.datetime.now()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for epoch in range(args.epochs):
        # 遺伝子型の選択（対戦相手）
        li_i_comp = [i_win]
        li_i_comp.extend(np.random.choice(
            a=list(set(range(args.n_geopath)) - set([i_win])),
            size=args.n_comp - 1))

        # それぞれのモデルを1epoch学習
        histories = []
        tmp_weights = []
        tmp_weights_save = []
        for i, i_comp in enumerate(li_i_comp):
            print('%s th, geopath%s is under training' % (epoch, i_comp))

            # モデルの作成，GPU並列
            model = pathnet.gene2model(li_geopath[i_comp])

            # compile
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            history = model.fit(
                x=train_generator,
                initial_epoch=0,
                epochs=args.geopath_epochs,
                verbose=1,
                validation_data=validation_generator,
                # callbacks=[EarlyStopping(patience=patience),
                #     ModelCheckpoint("models/models.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=False,
                #                     save_weights_only=False),
                #     tf.keras.callbacks.TensorBoard(
                #         log_dir='./logs',
                #         profile_batch='100,150',
                #         histogram_freq=1, write_grads=True, write_images=1,
                #         embeddings_freq=1)
                # ]
            )

            # 学習結果の格納
            history = np.array([
                [history.history['mae'][-1], history.history['loss'][-1], history.history['val_mae'][-1],
                 history.history['val_loss'][-1]]
            ])
            histories.append(copy.deepcopy(history))

            # 重みの格納
            tmp_weights.append(pathnet.extract_weights(model))
            tmp_weights_save.append(model.get_weights())

        # 結果の集計
        tmp_loss = [history[0, 3] for history in histories]
        which_win = np.argmin(tmp_loss)
        i_win = li_i_comp[which_win]

        if tmp_loss[which_win] <= best_val_loss:
            best_val_loss = tmp_loss[which_win]
            model.set_weights(tmp_weights_save[which_win])
            model.save(os.path.join(args.save_dir, "models.{}.hdf5".format(now.strftime('%Y%m%d_%H%M%S'))))

        print('%s th i_comp:%s geopath:%s acc: %s' % (epoch, li_i_comp, li_geopath[i_win], tmp_loss[which_win]))

        # 勝者による上書き ・突然変異
        for i in li_i_comp:
            if i != i_win:
                li_geopath[i] = pathnet.mutate_geopath(np.copy(li_geopath[i_win]))
        pathnet.adjustGeopath(li_geopath)

        # 重みの保存
        # I think should delete this code
        if args.do_original:
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
    df_data = pd.DataFrame(data, columns=['mae', 'loss', 'val_mae', 'val_loss'])
    df_data['geopath_best'] = [str(x) for x in geopath_best_log]
    df_data['geopath_set'] = [str(x) for x in geopath_set_log]
    df_data['i_comp'] = [str(x) for x in i_comp_log]
    df_data['i_win'] = i_win_log

    if args.do_original:
        s = "original"
    else:
        s = "proposed"

    now = datetime.datetime.now()
    epoch_log_name = 'swpathnet_%s-tournament-%s.csv' % (s, now.strftime('%Y%m%d_%H%M%S'))

    df_data.to_csv(os.path.join(args.save_dir, epoch_log_name), header=True)

    # model.set_weights(tmp_weights_save[which_win])
    # model.save(os.path.join(args.save_dir, "model_{}_{}.hdf5".format(s, now.strftime('%Y%m%d_%H%M%S'))))

    # TF 2.3 and earlier Keras puts all model construction in the same global background graph workspace,
    # which leads to a memory leak unless you explicitly call keras.backend.clear_session.
    tf.keras.backend.clear_session()


def get_parser():
    parser = argparse.ArgumentParser()

    # データセットのディレクトリ
    parser.add_argument('dataset_dir')

    # ログの保存先ディレクトリ
    parser.add_argument('save_dir')

    # 学習データの画像数
    #   CIFAR10: 50000, CIFAR100: 50000
    parser.add_argument('--num_images_train', type=int, default=50000, help='CIFAR10: 50000, CIFAR100: 50000')

    # テストデータの画像数
    #   CIFAR10: 10000, CIFAR100: 10000
    parser.add_argument('--num_images_test', type=int, default=10000, help='CIFAR10: 10000, CIFAR100: 10000')

    # 画像サイズ
    parser.add_argument('--image_size', type=int, default=224)

    # 遺伝子型の数
    parser.add_argument('--n_geopath', type=int, default=20)

    # 世代数
    parser.add_argument('--epochs', type=int, default=30)

    # 勝負する遺伝子型の数
    parser.add_argument('--n_comp', type=int, default=2)

    # バッチサイズ
    parser.add_argument('--batch_size', type=int, default=16)

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
    parser.add_argument('--dont_augment', action='store_false')

    # 層の範囲の選択
    #   True: 全て（含むBN）, False: CNN, FCのみ
    parser.add_argument('--transfer_all', action='store_true')

    # 学習可能層の初期値に学習済みパラメータを使う
    parser.add_argument('--finetune', action='store_true')

    # （テスト用）学習済みモデルの設定
    parser.add_argument('--trained_model', default=None)

    # 使う学習済みモデル
    parser.add_argument('--model_name', default='vgg16',
                        help='name of pre-trained network. this is disable by giving --trained_model')
    # Flag for original or proposed
    parser.add_argument('--do_original', action='store_true')

    return parser


# 引数の読み込み
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print('----PARSED ARGS----\n%s\n-----------------' % args)

    main(args)
