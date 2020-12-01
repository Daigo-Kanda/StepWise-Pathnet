import argparse
import datetime
import os

import tensorflow as tf
from tensorflow import keras

import ITrackerData_Person as data_gen


def main(args):
    tf.debugging.set_log_device_placement(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # 画像サイズの設定
    image_shape = (args.image_size, args.image_size, 3)

    base_model = keras.models.load_model(args.trained_model)

    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())

    a = model.get_weights()

    # 重み固定の有無
    if args.fix_pretrained:
        for layer in base_model.layers:
            layer.trainable = False

    # top layerをターゲットタスクに合わせる
    # if args.top_layer == 'add':
    #     #   https://gist.github.com/didacroyo/839bd1dbb67463df8ba8fb14eb3fde0c より
    #     # add a global spatial average pooling layer
    #     x = base_model.output
    #     x = keras.layers.GlobalAveragePooling2D()(x)
    #     # let's add a fully-connected layer
    #     x = keras.layers.Dense(1024, activation='relu')(x)
    #     # and a logistic layer -- let's say we have 200 classes
    #     predictions = keras.layers.Dense(args.num_classes, activation='softmax')(x)
    #     model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    # elif args.top_layer == 'replace':
    #     x = base_model.output
    #     x = keras.layers.GlobalAveragePooling2D()(x)
    #     predictions = keras.layers.Dense(args.num_classes, activation='softmax')(x)
    #     model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    # else:
    #     print('in valid --top_layer: %s' % args.top_layer)
    #     return -1

    print(model.summary())

    # generator setting
    train_generator = data_gen.ITrackerData(args.dataset_dir, 'train', (args.image_size, args.image_size), (25, 25),
                                            args.batch_size)
    test_generator = data_gen.ITrackerData(args.dataset_dir, 'test', (args.image_size, args.image_size), (25, 25),
                                           args.batch_size)

    # compile
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    b = model.get_weights()

    now = datetime.datetime.now()

    # callbacks
    cbks = [keras.callbacks.CSVLogger(
        os.path.join(args.save_dir, 'finetuning_%s_%s.csv' % ("eye_tracking", now.strftime('%Y%m%d_%H%M%S'))))]

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        verbose=1,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=cbks,
    )

    model.save('./model_finetuning_{}.hdf5'.format(now.strftime('%Y%m%d_%H%M%S')))

    ## save history
    # df_history = pd.DataFrame(history.history)
    # df_history.to_csv(os.path.join(args.save_dir, 'finetuning_%s_%s.csv' % (args.model_name, args.learning_number)))


def getParser():
    parser = argparse.ArgumentParser()

    # N回回す実験用のアレ
    parser.add_argument('learning_number', help='for sequential experiment')

    # データセットのディレクトリ
    parser.add_argument('dataset_dir')

    # ログの保存先ディレクトリ
    parser.add_argument('save_dir')

    # クラス数
    parser.add_argument('--num_classes', type=int, default=100)

    # 学習データの画像数
    #   CIFAR10: 50000, CIFAR100: 50000
    parser.add_argument('--num_images_train', type=int, default=50000, help='CIFAR10: 50000, CIFAR100: 50000')

    # テストデータの画像数
    #   CIFAR10: 10000, CIFAR100: 10000
    parser.add_argument('--num_images_test', type=int, default=10000, help='CIFAR10: 10000, CIFAR100: 10000')

    # 画像サイズ
    parser.add_argument('--image_size', type=int, default=224)

    # バッチサイズ
    parser.add_argument('--batch_size', type=int, default=16)

    # エポック
    parser.add_argument('--epochs', type=int, default=60)

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

    # （テスト用）学習済みモデルの設定
    parser.add_argument('--trained_model', default=None)

    # 使う学習済みモデル
    parser.add_argument('--model_name', default='vgg16',
                        help='name of pre-trained network. this is disable by giving --trained_model')

    # 識別層の扱い
    parser.add_argument('--top_layer', default='replace', help='method for classification layer. (add, replace)')

    # 重み固定の有無
    parser.add_argument('--fix_pretrained', action='store_true')

    return parser


# 引数の読み込み
if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    print('----PARSED ARGS----\n%s\n-----------------' % args)

    main(args)
