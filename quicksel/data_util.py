import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train_rows = np.load("/home/jitao/experiment/data/range_feature_data/cover_width_6000_attr_six_rows.npy").astype(np.float)

train_range = np.load("/home/jitao/experiment/data/range_feature_data/cover_width_6000_attr_six_features.npy").astype(
    np.float
)

# train_range = train_range[train_rows >= 1]
# train_rows = train_rows[train_rows >= 1]

sc = MinMaxScaler(feature_range=(0, 1))

MAX_CARDINALITY = 581012

row_range = train_rows / MAX_CARDINALITY
print(np.max(train_rows))
sc_range = sc.fit_transform(train_range)
row_range = np.expand_dims(row_range, axis=1)

data = np.concatenate((sc_range, row_range), axis=1)
print(np.max(sc_range, axis=-2))

print(np.min(sc_range, axis=-2))

print(np.max(row_range))

print(np.min(row_range))

with open("data/range_row_quicksel/cover_width_6d_range.txt", "w") as f:
    f.writelines(
        [
            "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.8f}\n".format(
                i[0],
                i[1],
                i[2],
                i[3],
                i[4],
                i[5],
                i[6],
                i[7],
                i[8],
                i[9],
                i[10],
                i[11],
                i[12],
                # i[13],
                # i[14],
                # i[15],
                # i[16],
                # i[17],
                # i[18],
                # i[19],
                # i[20]
                # i[18],
            )
            for i in data
        ]
    )
