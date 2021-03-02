import csv
import glob
import math
import time
from os.path import join

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import ITrackerData_person_tensor as id


# Grad_CAM全般の処理を行うクラス
class grad_cam():

    # construct
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model_path = model_path

        # layers name
        # this is magic number, have to change according to your code
        self.layer_name_face = 'conv2d_11'
        self.layer_name_right = 'conv2d_3'
        self.layer_name_left = 'conv2d_7'

        batch_size = 256

        # generate data
        generator = id.getData(batch_size=0, dataset_path=dataset_path, memory_size=150)

    # 出力を決定する関数（内部でしか使用しない）
    def decideOutput(self, model, i, true):
        if i == 0:
            return model.output[:, 0]
        if i == 1:
            return model.output[:, 1]
        if i == 2:
            return model.output[:, :]
        if i == 3:
            return 1 / tf.math.sqrt((model.output[:, 0] - true[0]) ** 2 + (model.output[:, 1] - true[1]) ** 2)

    # 提案手法を行う関数
    def gradcam_regression(self):

        # load model
        model = tf.keras.models.load_model(self.model_path)
        model.summary()

        # seqs = sorted(glob.glob(join(img_path, "0*")))

        # 画像を入力して各種出力を得る関数ライクなModel
        grad_model = tf.keras.models.Model([model.input],
                                           [model.get_layer(self.layer_name_right).output,
                                            model.get_layer(self.layer_name_left).output,
                                            model.get_layer(self.layer_name_face).output,
                                            model.output])

        # 個人の画像ディレクトリへアクセス
        for b, dir1 in enumerate(seqs):

            seqs2 = sorted(glob.glob(join(dir1, "0*")))

            # 一つのディレクトリごとにgrad_camを適応
            for h, dir in enumerate(seqs2):

                print(dir)

                # パスから画像を読み込み
                img_path_face = dir + "/face.png"
                # print(img_path_face)
                img_path_right = dir + "/right.png"
                img_path_left = dir + "/left.png"
                img_path_grid = dir + "/grid.png"

                # モデルで読み込める形式に変換
                img_face = image.load_img(img_path_face, target_size=(128, 128))
                img_face = image.img_to_array(img_face)
                img_face = np.expand_dims(img_face, axis=0)
                img_face = image_normalization(img_face)

                img_right = image.load_img(img_path_right, target_size=(128, 128))
                img_right = image.img_to_array(img_right)
                img_right = np.expand_dims(img_right, axis=0)
                img_right = image_normalization(img_right)

                img_left = image.load_img(img_path_left, target_size=(128, 128))
                img_left = image.img_to_array(img_left)
                img_left = np.expand_dims(img_left, axis=0)
                img_left = image_normalization(img_left)

                img_grid = image.load_img(img_path_grid, target_size=(25, 25))
                img_grid = image.img_to_array(img_grid)
                img_grid = np.delete(img_grid, [0, 1], 2)
                img_grid = np.where(img_grid > 0, 1, 0)
                img_grid = np.expand_dims(img_grid, axis=0)

                # csvから真の値を取得
                with open("{}/data.csv".format(dir), "r", encoding="utf_8", errors="", newline="") as csv_file:
                    # リスト形式
                    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                                   skipinitialspace=True)
                    header = next(f)
                    true_x = float(header[0])
                    true_y = float(header[1])

                # xのみ，yのみ，x+y，customの順番で実行
                # x:0 y:1 x+y:2 custom:3
                for i in range(4):

                    # 目の場合と顔の場合わけ
                    # 目_right:0 目_left:1 顔:2
                    for j in range(3):

                        second = time.time()

                        time_grad_start = time.time()

                        # conv_outputs : 特定の畳み込みそうの出力
                        # custom_outputs : カスタムした出力
                        # last_outputs : 推定値

                        with tf.GradientTape() as tape:
                            conv_outputs_right, conv_outputs_left, conv_outputs_face, last_outputs = grad_model(
                                [img_right, img_left, img_face, img_grid])

                            if j == 0:
                                x = conv_outputs_right
                                img = cv2.imread(img_path_right)
                            if j == 1:
                                img = cv2.imread(img_path_left)
                                x = conv_outputs_left
                            if j == 2:
                                img = cv2.imread(img_path_face)
                                x = conv_outputs_face

                            if i == 0:
                                y = last_outputs[:, 0]
                            if i == 1:
                                y = last_outputs[:, 1]
                            if i == 2:
                                y = last_outputs
                            if i == 3:
                                y = 1 / tf.math.sqrt(
                                    (last_outputs[:, 0] - true_x) ** 2 + (last_outputs[:, 1] - true_y) ** 2)

                        grads = tape.gradient(y, x)[0]

                        time_grad_end = time.time()

                        # print(time_grad_end - time_grad_start)

                        outputs = x[0]

                        weights = tf.reduce_mean(grads, axis=(0, 1))

                        cam = np.zeros(outputs.shape[0:2])

                        for k, w in enumerate(weights):
                            cam += w * outputs[:, :, k]

                        heatmap = np.maximum(cam, 0)
                        heatmap /= np.max(heatmap)

                        heatmap = cv2.resize(heatmap, (128, 128))
                        heatmap = np.uint8(255 * heatmap)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                        # heatmapの保存
                        cv2.imwrite("{}/{}_{}_heatmap.png".format(dir, j, i), heatmap)

                        superimposed_img = heatmap * 0.4 + img
                        cv2.imwrite("{}/{}_{}.png".format(dir, j, i), superimposed_img)

                        elapsed_time = time.time()
                        print("elapsed_time3:{}[sec]".format(elapsed_time - second))

                        if i == 2 and j == 0:
                            # print(output_value[:,0])
                            # print(output_value[:,1])
                            with open(dir1 + "/length.csv", "a") as f:
                                writer = csv.writer(f)
                                writer.writerow(
                                    [dir[-5:],
                                     math.sqrt(
                                         (last_outputs[:, 0] - true_x) ** 2 + (last_outputs[:, 1] - true_y) ** 2)])

            snapshot2 = tracemalloc.take_snapshot()

            top_status = snapshot2.compare_to(snapshot1, 'lineno')

            print("[ Top 10 differences ]")
            for stat in top_status[:10]:
                print(stat)

            # print(gc.get_stats()[2])
            # gc.collect()
            # print(gc.get_stats()[2])

    # 提案手法を行う関数
    # 最終畳み込みそうのnpzを取得する
    def regression_get_npz(self, model_path, img_path):

        tracemalloc.start()

        # 各レイヤーの名前
        layer_name_face = 'conv2d_11'
        layer_name_right = 'conv2d_3'
        layer_name_left = 'conv2d_7'

        face_dir = "/mnt/data2/img/20200209/face"
        right_dir = "/mnt/data2/img/20200209/right_eye"
        left_dir = "/mnt/data2/img/20200209/left_eye"
        grid_dir = "/mnt/data2/img/20200209/grid"

        # モデルの読み込み
        model = tf.keras.models.load_model(model_path)
        seqs_face = sorted(glob.glob(join(face_dir, "*")))
        seqs_right = sorted(glob.glob(join(right_dir, "*")))
        seqs_left = sorted(glob.glob(join(left_dir, "*")))
        seqs_grid = sorted(glob.glob(join(grid_dir, "*")))

        # 画像を入力して各種出力を得る関数ライクなModel
        grad_model = tf.keras.models.Model([model.input],
                                           [model.get_layer(layer_name_right).output,
                                            model.get_layer(layer_name_left).output,
                                            model.get_layer(layer_name_face).output,
                                            model.output])

        true_lists = []

        # csvから真の値を取得
        with open("{}/data.csv".format(img_path), "r", encoding="utf_8", errors="", newline="") as csv_file:
            # リスト形式
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                           skipinitialspace=True)
            for row in f:
                true_lists.append(row)

        # データ保存用のndarray
        # frame数，画像のタイプ，使用するメソッド，画像サイズｘ，画像サイズｙ
        npz = np.zeros([len(seqs_face), 3, 4, 12, 12])

        # 個人の画像ディレクトリへアクセス
        for h, (face, right, left, grid, true_list) in enumerate(
                zip(seqs_face, seqs_right, seqs_left, seqs_grid, true_lists)):

            start = time.time()

            # モデルで読み込める形式に変換
            img_face = image.load_img(face, target_size=(128, 128))
            img_face = image.img_to_array(img_face)
            img_face = np.expand_dims(img_face, axis=0)
            img_face = image_normalization(img_face)

            img_right = image.load_img(right, target_size=(128, 128))
            img_right = image.img_to_array(img_right)
            img_right = np.expand_dims(img_right, axis=0)
            img_right = image_normalization(img_right)

            img_left = image.load_img(left, target_size=(128, 128))
            img_left = image.img_to_array(img_left)
            img_left = np.expand_dims(img_left, axis=0)
            img_left = image_normalization(img_left)

            img_grid = image.load_img(grid, target_size=(25, 25))
            img_grid = image.img_to_array(img_grid)
            img_grid = np.delete(img_grid, [0, 1], 2)
            img_grid = np.where(img_grid > 0, 1, 0)
            img_grid = np.expand_dims(img_grid, axis=0)

            true_x = float(true_list[0])
            true_y = float(true_list[1])

            # xのみ，yのみ，x+y，customの順番で実行
            # x:0 y:1 x+y:2 custom:3
            for i in range(4):

                # 目の場合と顔の場合わけ
                # 目_right:0 目_left:1 顔:2
                for j in range(3):

                    second = time.time()

                    time_grad_start = time.time()

                    # conv_outputs : 特定の畳み込みそうの出力
                    # custom_outputs : カスタムした出力
                    # last_outputs : 推定値

                    with tf.GradientTape() as tape:
                        conv_outputs_right, conv_outputs_left, conv_outputs_face, last_outputs = grad_model(
                            [img_right, img_left, img_face, img_grid])

                        if j == 0:
                            x = conv_outputs_right
                            img = cv2.imread(right)
                        if j == 1:
                            img = cv2.imread(left)
                            x = conv_outputs_left
                        if j == 2:
                            img = cv2.imread(face)
                            x = conv_outputs_face

                        if i == 0:
                            y = last_outputs[:, 0]
                        if i == 1:
                            y = last_outputs[:, 1]
                        if i == 2:
                            y = last_outputs
                        if i == 3:
                            y = 1 / tf.math.sqrt(
                                (last_outputs[:, 0] - true_x) ** 2 + (last_outputs[:, 1] - true_y) ** 2)

                    grads = tape.gradient(y, x)[0]

                    time_grad_end = time.time()

                    # print(time_grad_end - time_grad_start)

                    outputs = x[0]

                    weights = tf.reduce_mean(grads, axis=(0, 1))

                    cam = np.zeros(outputs.shape[0:2])

                    for k, w in enumerate(weights):
                        cam += w * outputs[:, :, k]

                    heatmap = np.maximum(cam, 0)
                    if np.max(heatmap) != 0:
                        heatmap /= np.max(heatmap)
                    else:
                        print("heat map is empty!!!")

                    npz[h, j, i, :, :] = np.copy(heatmap)

            print("{} : {}".format(h, time.time() - start))

        np.save(img_path + "/grad_cam_heatmap", npz)
