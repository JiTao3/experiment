from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import pandas as pd
import os
from utils import log_transform, computeRangeCardinality, buildEqualDepthHist

MAX_CARDINALITY = 581012
size = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
columns = [2, 3, 4, 5, 6]
COLS = ["two", "three", "four", "five", "six"]

# size = [100, 200, 300, 400]
# columns = [2, 3]
# COLS = ["two", "three"]


def ebo_trans(a):
    return np.power(a[0], 0.125) * np.power(a[1], 0.25) * np.power(a[2], 0.5) * a[3]


for col, c in enumerate(columns):
    for j, s in enumerate(size):
        print(c, s)
        range_rows = np.load(
            "/home/jitao/experiment/data/range_feature_data/cover_5000_attr_{}_rows.npy".format(COLS[col])
        ).astype(np.float32)
        range_source = np.load(
            "/home/jitao/experiment/data/range_feature_data/cover_5000_attr_{}_range_features.npy".format(COLS[col])
        ).astype(np.float32)
        range_rows = log_transform(range_rows)
        # selectivity feature
        raw_data = pd.read_csv("cover.csv").sample(30000).to_numpy()
        bins = []
        bin_size = []
        for i in range(c):
            b, bs = buildEqualDepthHist(raw_data[:, i], nbin=100)
            bins.append(b)
            bin_size.append(bs)

        selectivity = []
        for each in range_source:
            sel = []
            for k in range(0, c):
                sel.append(computeRangeCardinality(bins[k], bin_size[k], each[2 * k], each[2 * k + 1]))
            selectivity.append(sel)
        selectivity = np.array(selectivity)
        selectivity[selectivity == 0] = 1
        selectivity = selectivity / 581012

        AVI_feature = selectivity
        if col >= 4:
            EBO_source = np.apply_along_axis(sorted, 1, np.partition(selectivity, 4)[:, :4])
            EBO_feature = np.apply_along_axis(ebo_trans, 1, EBO_source).reshape((-1, 1))
        else:
            pass
        Minsel_feature = selectivity.min(axis=1).reshape((-1, 1))

        sc = MinMaxScaler((0, 1000))
        range_feature = sc.fit_transform(range_source)
        if col >= 4:
            data = np.concatenate(
                (range_feature, np.log(AVI_feature), np.log(EBO_feature), np.log(Minsel_feature)), axis=1
            )
        else:
            data = np.concatenate((range_feature, np.log(AVI_feature), np.log(Minsel_feature)), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(data, range_rows, test_size=1000, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=s, random_state=1)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        # param = { 'max_leaves':16,'max_depth':5,'objective': 'reg:squarederror','gpu_id':1,"tree_method":"gpu_hist"}
        param = {"max_leaves": 16, "max_depth": 5, "objective": "reg:squarederror"}

        bst = xgb.XGBRegressor(**param)
        bst.fit(X_train, y_train)
        num_round = 500
        # evallist = [(dtest, 'eval'), (dtrain, 'train')]
        bst = xgb.train(param, dtrain, num_round)
        pred = bst.predict(dtest)

        bst.save_model("/home/jitao/experiment/model/xgb/{}_{}.json".format(c, s))

        print("dimension: ", c)
        print("train size: ", s)
        print("XGB rmse ", np.sqrt(mean_squared_error(np.exp2(pred) / 581012, np.exp2(y_test) / 581012)))

    # 0.015312372231701224
