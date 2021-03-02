import csv
import glob
import math

import numpy as np
import pandas as pd

# data path
# kenkyusitu
# paths = glob.glob("/kanda_tmp/StepWise-Pathnet/stochastic/test_results/*")
# kanda
path = "/mnt/data2/StepWise_PathNet/Results/201222/scratch/test_results/"
paths = glob.glob(path + "*")
save_path = path + "data_test.csv"
# csv_files = glob.glob("/mnt/data2/StepWise_PathNet/Test/Proposed/*.csv")

mse_ave_list = []
mae_ave_list = []

mse_list = []
mae_list = []

for path in paths:
    df = pd.read_csv(path)
    mse_list.append(df["test_loss"].mean())
    mae_list.append(df["test_mae"].mean())

mae_ave = np.nanmean(mae_list)
mse_ave = np.nanmean(mse_list)

print(mse_ave)

with open(save_path, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(['test_mse', 'test_mae'])
    for mse, mae in zip(mse_list, mae_list):
        if not math.isnan(mse) or math.isnan(mae):
            writer.writerow([mse, mae])

# with open(save_path, 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(train_ave_list)
#     writer.writerow(test_ave_list)
# writer.writerow([np.mean(train_ave_list), np.mean(test_ave_list)])
