import glob

import numpy as np
import pandas as pd

csv_files = glob.glob("/mnt/data2/StepWise_PathNet/Test/Proposed/*.csv")

# mae list
val_mae_list = []

# 読み込むファイルのリストを表示
for a in csv_files:
    print(a)
    df = pd.read_csv(a)
    val_mae_list.append([df["val_mae"].min(), a])

average_val_mae = np.mean([mae[0] for mae in val_mae_list])

print([mae[0] for mae in val_mae_list])
print(average_val_mae)
