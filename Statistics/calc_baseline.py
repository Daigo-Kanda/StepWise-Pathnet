import csv
import glob

import numpy as np
import pandas as pd

# data path
paths = glob.glob("/home/daigokanda/PycharmProjects/StepWise-Pathnet/baseline/*")
save_path = "data_baseline.csv"
# csv_files = glob.glob("/mnt/data2/StepWise_PathNet/Test/Proposed/*.csv")

train_ave_list = []
test_ave_list = []

for path in paths:
    csv_files = glob.glob(path + "/*.csv")
    train_ave_list = []
    test_ave_list = []

    for file in csv_files:
        if 'train' in file:
            print(file)
            df = pd.read_csv(file)
            train_ave_list.append(df["mean_absolute_error"].mean())
        elif 'val' in file:
            print(file)
            df = pd.read_csv(file)
            test_ave_list.append(df["mean_absolute_error"].mean())

    with open(save_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(train_ave_list + test_ave_list)

# with open(save_path, 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(train_ave_list)
#     writer.writerow(test_ave_list)
    # writer.writerow([np.mean(train_ave_list), np.mean(test_ave_list)])
