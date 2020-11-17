from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime
import pandas as pd
import os
# from utils import log_transform, computeRangeCardinality, buildEqualDepthHist


MAX_CARDINALITY = 581012
size = [200, 400, 600, 800, 1000]
columns = [2, 3, 4, 5, 6]
COLS = ["two", "three", "four", "five", "six"]
columns = [6]
COLS = ["six"]


def execute_query(data, query, dimension):
    if dimension == 2:
        df = data[(data[6] >= query[0]) & (data[6] <= query[1]) & (data[8] >= query[2]) & (data[8] <= query[3])]
    if dimension == 3:
        df = data[
            (data[0] >= query[0])
            & (data[0] <= query[1])
            & (data[5] >= query[2])
            & (data[5] <= query[3])
            & (data[9] >= query[4])
            & (data[9] <= query[5])
        ]
    if dimension == 4:
        df = data[
            (data[0] >= query[0])
            & (data[0] <= query[1])
            & (data[1] >= query[2])
            & (data[1] <= query[3])
            & (data[2] >= query[4])
            & (data[2] <= query[5])
            & (data[3] >= query[6])
            & (data[3] <= query[7])
        ]
    if dimension == 5:
        df = data[
            (data[0] >= query[0])
            & (data[0] <= query[1])
            & (data[1] >= query[2])
            & (data[1] <= query[3])
            & (data[2] >= query[4])
            & (data[2] <= query[5])
            & (data[3] >= query[6])
            & (data[3] <= query[7])
            & (data[4] >= query[8])
            & (data[4] <= query[9])
        ]
    if dimension == 6:
        df = data[
            (data[0] >= query[0])
            & (data[0] <= query[1])
            & (data[1] >= query[2])
            & (data[1] <= query[3])
            & (data[2] >= query[4])
            & (data[2] <= query[5])
            & (data[3] >= query[6])
            & (data[3] <= query[7])
            & (data[4] >= query[8])
            & (data[4] <= query[9])
            & (data[5] >= query[10])
            & (data[5] <= query[11])
        ]

    return len(df)


for col, c in enumerate(columns):
    for j, s in enumerate(size):
        range_rows = np.load("../range_feature_data/cover_grid_5000_attr_{}_rows.npy".format(COLS[col])).astype(
            np.float32
        )
        range_source = np.load(
            "../range_feature_data/cover_grid_5000_attr_{}_range_features.npy".format(COLS[col])
        ).astype(np.float32)
        range_rows = range_rows / 581012
        # selectivity feature
        raw_data = pd.read_csv("../data/cover.csv", names=range(0, 10)).sample(s)
        # raw_data = pd.read_csv("/data/sunluming/datasets/cover.csv").to_numpy()
        # print(raw_data)

        cardinality = []
        train_starttime = datetime.datetime.now()
        for each in range_source:
            # print(each)
            sel = execute_query(raw_data, each, c)
            cardinality.append(sel)
        cardinality = np.array(cardinality)
        selectivity = cardinality / s
        # print(selectivity[:20])
        # print(range_rows[:20])
        train_endtime = datetime.datetime.now()
        print("dimension: ", c)
        print("sample number: ", s)
        print("sampling rmse ", np.sqrt(mean_squared_error(selectivity, range_rows)))
        print("training time: ", (train_endtime - train_starttime))
        print("********************")
