import csv
import glob

import numpy as np
import pandas as pd
from operator import add, truediv

# data path
paths = glob.glob("/mnt/data2/StepWise_PathNet/Results/201216/stepwise_original/*")
save_path = "data_original.csv"
# csv_files = glob.glob("/mnt/data2/StepWise_PathNet/Test/Proposed/*.csv")

train_ave_list = []
test_ave_list = []

path_sum = [0] * 25
count = 0
for path in paths:
    csv_files = glob.glob(path + "/*.csv")
    train_list = []
    test_list = []

    for file in csv_files:

        print(file)
        df = pd.read_csv(file)
        count += len(df.index)
        for best_path in df["geopath_best"]:
            path_sum = map(add, path_sum, [int(s) for s in best_path.strip('[]').split()])

        # b = df["geopath_best"][0].strip('[]')
        # a = b.split()
        # test_list.append(df["geopath_best"])
        # train_list.append(df["mae"][df["val_mae"].idxmin()])

    # test_ave_list.append(np.mean(test_list))
    # train_ave_list.append(np.mean(train_list))
    #
    # with open(save_path, 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(train_list + test_list)
# lists = list(path_sum)
final = list(map(lambda x: (x / count), path_sum))
print(final)

# with open(save_path, 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(train_ave_list)
#     writer.writerow(test_ave_list)
#     writer.writerow([np.mean(train_ave_list), np.mean(test_ave_list)])
