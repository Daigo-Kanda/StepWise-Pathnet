import csv
import glob

import numpy as np
import pandas as pd

# data path
paths = glob.glob("/mnt/data2/StepWise_PathNet/Results/201222/stepwise_new/*")
save_path = "data_stepwise.csv"
# csv_files = glob.glob("/mnt/data2/StepWise_PathNet/Test/Proposed/*.csv")

with open(save_path, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(['train', 'validation'])

for path in paths:
    csv_files = glob.glob(path + "/best*")
    train_list = []
    test_list = []
    train_ave_list = []
    test_ave_list = []
    if not 'test_results' in path:
        for file in csv_files:
            print(file)
            df = pd.read_csv(file)
            test_list.append(df["val_loss"].min())
            train_list.append(df["loss"][df["val_loss"].idxmin()])

        test_ave_list.append(np.mean(test_list))
        train_ave_list.append(np.mean(train_list))

        with open(save_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(train_ave_list + test_ave_list)

# with open(save_path, 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(train_ave_list)
#     writer.writerow(test_ave_list)
    # writer.writerow([np.mean(train_ave_list), np.mean(test_ave_list)])
