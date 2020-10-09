import numpy as np


def split_data(data, label, num_part):
    data = np.split(data, num_part, axis=0)
    label = np.split(label, num_part, axis=0)
    return data, label


def random_from_data(data, label, size):
    idx = np.random.choice(np.arange(len(data)), int(size), replace=False)
    data_sample = data[idx]
    label_sample = label[idx]
    return data_sample, label_sample

