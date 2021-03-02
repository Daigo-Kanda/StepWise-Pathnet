import glob
import os

import tensorflow as tf

import ITrackerData_person_tensor as ds
import stochastic_transfer_eye_tracker

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000),
        #      tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# hyper parameter
# dataset_path = "/kanda_tmp/GazeCapture_pre"
# kanda
# dataset_path = "/mnt/data/DataSet/GazeCapture_pre"
# matsuura
dataset_path = "/home/kanda/GazeCapture_pre"
model_path = "model/models.046-2.46558.hdf5"
participants_num = 30
loop_num = 5
batch_size = "64"
image_size = "224"

participants_path = glob.glob(os.path.join(dataset_path, "**"))

participants_count = []
k = 0
for i, participant_path in enumerate(participants_path):
    metaFile = os.path.join(participant_path, 'metadata_person.mat')

    if os.path.exists(metaFile):
        participants_count.append(len(ds.loadMetadata(metaFile)['frameIndex']))
    else:
        participants_count.append(0)

tmp = zip(participants_count, participants_path)

# sorting
sorted_tmp = sorted(tmp, reverse=True)
participants_count, participants_path = zip(*sorted_tmp)

for i in range(participants_num):
    for j in range(loop_num):
        # parser = sw_pathnetmod_tournament_eye_tracker.get_parser()
        # sw_pathnetmod_tournament_eye_tracker.main(parser.parse_args(
        #     [participants_path[i], "./my_stepwise/{}".format(participants_path[i][-5:]), "--image_size", image_size,
        #      "--batch_size", batch_size,
        #      "--epochs", "100", "--trained_model", model_path, "--transfer_all"])
        # )

        parser = stochastic_transfer_eye_tracker.get_parser()
        stochastic_transfer_eye_tracker.main(parser.parse_args(
            [participants_path[i], "./stochastic/{}".format(participants_path[i][-5:]), "--image_size",
             image_size, "--batch_size", batch_size,
             "--epochs", "30", "--trained_model", model_path, "--transfer_all", "--do_original", '--n_geopath', '100'])
        )
