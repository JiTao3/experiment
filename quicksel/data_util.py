import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train_rows = np.load("/home/jitao/experiment/data/random_5000_attr_059_rows.npy").astype(np.float)

train_range = np.load("/home/jitao/experiment/data/random_5000_attr_059_features.npy").astype(np.float)

train_range = train_range[train_rows >= 2]
train_rows = train_rows[train_rows >= 2]

sc = MinMaxScaler(feature_range=(0, 1))

row_range = train_rows / np.max(train_rows)
print(np.max(train_rows))
sc_range = sc.fit_transform(train_range)
row_range = np.expand_dims(row_range, axis=1)

data = np.concatenate((sc_range, row_range), axis=1)
print(np.max(sc_range, axis=-2))

print(np.min(sc_range, axis=-2))

print(np.max(row_range))

print(np.min(row_range))

with open("data/range_row.txt", "w") as f:
    f.writelines(
        [
            "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.8f}\n".format(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
            for i in data
        ]
    )
