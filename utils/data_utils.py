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


def log_transform(data):
    data = np.array(data)
    return np.log2(data)


def selectativity_transform(data, max_cardinality):
    return data / max_cardinality


def bound_result(result, min_bound, max_bound):
    result = np.where(result < min_bound, min_bound, result)
    result = np.where(result > max_bound, max_bound, result)
    return result
