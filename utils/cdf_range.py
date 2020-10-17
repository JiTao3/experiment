import tensorflow as tf
import os

# from IPython.core.pylabtools import figsize
import copy
import logging
import numpy as np
import pandas as pd
import datetime
import sys
import tensorflow_lattice as tfl
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import feature_column as fc
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance


def cdf_range_3d(lattice_model, featuer):
    assert featuer.shape[-1] == 6
    x1l, x1h, x2l, x2h, x3l, x3h = (
        featuer[:, 0],
        featuer[:, 1],
        featuer[:, 2],
        featuer[:, 3],
        featuer[:, 4],
        featuer[:, 5],
    )
    xhhh = [x1h, x2h, x3h]
    xhhl = [x1h, x2h, x3l]
    xhlh = [x1h, x2l, x3h]
    xlhh = [x1l, x2h, x3h]
    xhll = [x1h, x2l, x3l]
    xlhl = [x1l, x2h, x3l]
    xllh = [x1l, x2l, x3h]
    xlll = [x1l, x2l, x3l]

    xhhh_p = lattice_model.predict(xhhh)
    xhhl_p = lattice_model.predict(xhhl)
    xhlh_p = lattice_model.predict(xhlh)
    xlhh_p = lattice_model.predict(xlhh)
    xhll_p = lattice_model.predict(xhll)
    xlhl_p = lattice_model.predict(xlhl)
    xllh_p = lattice_model.predict(xllh)
    xlll_p = lattice_model.predict(xlll)

    pres = xhhh_p - xhhl_p - xhlh_p - xlhh_p + xhll_p + xlhl_p + xllh_p - xlll_p

    return pres
