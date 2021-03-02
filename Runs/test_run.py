import csv
import glob
import os

import tensorflow as tf

import ITrackerData_person_tensor as ds
import test_eyetracker as tt

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

which_data = 'stepwise_new'
is_baseline = False

# if use random data, change test_eyetracker's data generator
use_random_data = False

# original model path
original_model_path = 'model/models.046-2.46558.hdf5'
# kenkyusitu
# dataset_path = "/kanda_tmp/GazeCapture_pre"
# kanda
dataset_path = "/mnt/data/DataSet/GazeCapture_pre"
# matsuura
# dataset_path = "/home/kanda/GazeCapture_pre"

# kenkyusitu
# participants_models_path = "/kanda_tmp/StepWise-Pathnet/" + which_data
# kanda
participants_models_path = "/mnt/data2/StepWise_PathNet/Results/201222/" + which_data

participants_num = 50
loop_num = 5
batch_size = "256"
image_size = "224"

participants_path = glob.glob(os.path.join(dataset_path, "**"))

participants_count = []
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
    histories = []

    if use_random_data:
        data_path = dataset_path
        random_str = 'random'

    else:
        data_path = participants_path[i]
        random_str = 'not_random'

    # k = abs(i - participants_num - 1)
    models_path = glob.glob(os.path.join(os.path.join(participants_models_path, participants_path[i][-5:]), '*.hdf5'))

    if is_baseline:
        parser = tt.get_parser()
        history = tt.main(parser.parse_args(
            [data_path, "./{}/{}".format(which_data, participants_path[i][-5:]), "--image_size",
             image_size, "--batch_size", batch_size,
             "--epochs", "100", "--trained_model", original_model_path])
        )
        histories.append(history)

        os.makedirs(os.path.join(participants_models_path, 'test_results'), exist_ok=True)
        file_name = os.path.join(os.path.join(participants_models_path, 'test_results'),
                                 'test_{}_{}.csv'.format(which_data, participants_path[i][-5:]))
    else:
        if len(models_path) != 0:
            for model_path in models_path:
                parser = tt.get_parser()
                history = tt.main(parser.parse_args(
                    [data_path, "./{}/{}".format(which_data, participants_path[i][-5:]), "--image_size",
                     image_size, "--batch_size", batch_size,
                     "--epochs", "100", "--trained_model", model_path])
                )
                histories.append(history)

            os.makedirs(os.path.join(participants_models_path, 'test_results'), exist_ok=True)
            file_name = os.path.join(os.path.join(participants_models_path, 'test_results'),
                                     'test_{}_{}.csv'.format(which_data, participants_path[i][-5:]))

    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['test_loss', 'test_mae'])
        for history in histories:
            writer.writerow(history)
