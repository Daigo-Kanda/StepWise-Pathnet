import tensorflow as tf
from tensorflow import keras

import ITrackerData_person_tensor as data_gen


def gene2model_aco(model, dic_path):
    for layer in model.layers:
        # 重みがあればgeneを参照してtrainableを変更
        #   新規学習レイヤーの場合は重みをロード
        if layer.name in dic_path:
            layer.trainable = bool(dic_path[layer.name])


def freeze_model(model):
    for layer in model.layers:
        if 'dense' not in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True


# hyper parameter
trained_model_path = "model/models.046-2.46558.hdf5"
dataset_dir = '/kanda_tmp/GazeCapture_pre/00804'

model = keras.models.load_model(trained_model_path)
fine_tuning_model = keras.models.load_model(trained_model_path)

x = model(model.inputs, training=False)
fine_tuning_model = tf.keras.Model(model.inputs, x)

fine_tuning_model.summary()

# generate data
data = data_gen.getData(batch_size=64, memory_size=150, dataset_path=dataset_dir)
train_generator = data[0]
validation_generator = data[1]
test_generator = data[2]

path = {"conv2d_4": 1, "batch_normalization_2": 0, "conv2d_5": 1, "batch_normalization_3": 1, "conv2d_6": 0,
        "conv2d_7": 0, "conv2d": 1, "batch_normalization": 0, "conv2d_1": 0, "batch_normalization_1": 0, "conv2d_2": 0,
        "conv2d_3": 1, "dense": 1, "conv2d_8": 0, "batch_normalization_4": 0, "conv2d_9": 1, "batch_normalization_5": 1,
        "conv2d_10": 0, "conv2d_11": 1, "dense_1": 1, "dense_2": 1, "dense_3": 1, "dense_4": 1, "dense_5": 1,
        "dense_6": 1}

# freeze_model(model)
#
# fine_tuning_model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
#
# # train just one epoch
# fine_tuning_model.fit(
#     x=train_generator,
#     initial_epoch=0,
#     epochs=20,
#     verbose=2,
#     validation_data=validation_generator,
# )

weights = fine_tuning_model.get_weights()

gene2model_aco(model, path)

fine_tuning_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='mse', metrics=['mae'])

fine_tuning_model.fit(
    x=train_generator,
    initial_epoch=0,
    epochs=30,
    verbose=2,
    validation_data=validation_generator,
)

fine_history = fine_tuning_model.evaluate(
    x=test_generator,
    verbose=2,
)

print(fine_history)

fine_tuning_model.set_weights(weights)

fine_tuning_model.trainable = True
model.trainable = True

fine_tuning_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='mse', metrics=['mae'])

fine_tuning_model.fit(
    x=train_generator,
    initial_epoch=0,
    epochs=30,
    verbose=2,
    validation_data=validation_generator,
)

fine_history = fine_tuning_model.evaluate(
    x=test_generator,
    verbose=2,
)

print(fine_history)

#
# model = gene2model_aco(model, path)
#
# model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='mse', metrics=['mae'])
#
# # train just one epoch
# history = model.fit(
#     x=train_generator,
#     initial_epoch=0,
#     epochs=30,
#     verbose=2,
#     validation_data=validation_generator,
# )
#
# evaluate_history = model.evaluate(
#     x=test_generator,
#     verbose=2,
# )
#
# print(evaluate_history)

#
# for i in range(cycle):
#     li_path = []
#     li_edge = []
#     li_loss = []
#     histories = []
#
#     # make path and that edge
#     for j in range(n_geopath):
#         tmp = baco.gen_path()
#         for k in tmp[0]:
#             if 'dense' in k:
#                 tmp[0][k] = 1
#         li_path.append(tmp[0])
#         li_edge.append(tmp[1])
#
#     start = time.time()
#
#     for k, batch in enumerate(train_generator):
#
#         if k == n_geopath:
#             break
#
#         get_model_time = time.time()
#         # make model from path data (set freeze or not)
#         model = pathnet.gene2model_aco(li_path[k])
#
#         # print(model.optimizer.get_weights())
#
#         print('model prepare need {} seconds'.format(time.time() - get_model_time))
#
#         fit_time = time.time()
#
#         # train just one epoch
#         history = model.fit(
#             x=batch[0],
#             y=batch[1],
#             batch_size=args.batch_size,
#             initial_epoch=0,
#             epochs=1,
#             verbose=2,
#             validation_data=validation_generator,
#             validation_batch_size=args.batch_size
#         )
#
#         print('fit time is {}'.format(time.time() - fit_time))
#
#         # results
#         history = np.array([
#             [history.history['mae'][-1], history.history['loss'][-1], history.history['val_mae'][-1],
#              history.history['val_loss'][-1]]
#         ])
#
#         # save acc
#         li_loss.append(history[0, 3])
#         histories.append(copy.deepcopy(history))
#
#         tf.keras.backend.clear_session()
#         # ガベコレ
#         del history, model, batch
#         gc.collect()
#
#     print('elapsed time is {}'.format(time.time() - start))
#
#     # 結果の集計
#     tmp_loss = [history[0, 3] for history in histories]
#     which_win = np.argmin(tmp_loss)
#
#     if tmp_loss[which_win] <= best_val_loss:
#         best_val_loss = tmp_loss[which_win]
#         best_path = li_path[which_win]
#         print(best_path)
#
#     # ログの格納
#     history_log.append(histories[which_win])
#     geopath_best_log.append(copy.deepcopy(li_path[which_win]))
#     # geopath_set_log.append(copy.deepcopy(li_path))
#
#     # update pheromone for next cycle
#     baco.update_pheromone(li_edge, li_loss)
#
#     # ガベコレ
#     del histories, li_path, li_edge, li_loss, tmp_loss, which_win
#     gc.collect()
#
# # ログデータの吐き出し
# data = np.array(history_log).reshape((cycle, 4))
# df_data = pd.DataFrame(data, columns=['mae', 'loss', 'val_mae', 'val_loss'])
# df_data['geopath_best'] = [json.dumps(x, default=myconverter) for x in geopath_best_log]
#
# now = datetime.datetime.now()
# epoch_log_name = 'swpathnet_%s-tournament-%s.csv' % ('baco', now.strftime('%Y%m%d_%H%M%S'))
#
# df_data.to_csv(os.path.join(args.save_dir, epoch_log_name), header=True)
#
# ####################################################################################################################
# # train best path
# model = pathnet.gene2model_aco(best_path)
#
# # generate data
# data = data_gen.getData(batch_size=args.batch_size, memory_size=150, dataset_path=args.dataset_dir)
# train_generator = data[0]
# validation_generator = data[1]
#
# now = datetime.datetime.now()
# # callbacks
# cbks = [keras.callbacks.CSVLogger(
#     os.path.join(args.save_dir, '%s_%s_%s.csv' % ('baco', "eye_tracking", now.strftime('%Y%m%d_%H%M%S')))),
#     ModelCheckpoint(
#         os.path.join(args.save_dir, "models.{}.hdf5".format(now.strftime('%Y%m%d_%H%M%S'))),
#         monitor='val_loss', save_best_only=True,
#         save_weights_only=False)
# ]
#
# model.fit(
#     x=train_generator,
#     initial_epoch=0,
#     epochs=args.epochs,
#     verbose=1,
#     validation_data=validation_generator,
#     callbacks=cbks,
# )
#
# ####################################################################################################################
#
# tf.keras.backend.clear_session()
