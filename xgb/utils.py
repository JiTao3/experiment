import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf

def print_qerror(pred, label):
    qerror = []
    for i in range(len(pred)):
        if pred[i]==0 and float(label[i])==0:
            qerror.append(1)
        elif pred[i]==0:
            qerror.append(label[i])
        elif label[i]==0:
            qerror.append(pred[i])
        elif pred[i] > float(label[i]):
            qerror.append(float(pred[i]) / float(label[i]))
        else:
            qerror.append(float(label[i]) / float(pred[i]))
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))

def print_qerror_zero(pred, label,cardinality):
    qerror = []
    for i in range(len(pred)):
        if pred[i]==0 and float(label[i])==0:
            qerror.append(1)
        elif pred[i]==0:
            qerror.append(label[i]*cardinality)
        elif label[i]==0:
            qerror.append(pred[i]*cardinality)
        elif pred[i] > float(label[i]):
            qerror.append(float(pred[i]) / float(label[i]))
        else:
            qerror.append(float(label[i]) / float(pred[i]))
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))

def log_transform(data):
    data[data==0] = 1
    data = np.array(data)
    return np.log2(data)

def compute_quantiles(features,
                      num_keypoints=10,
                      clip_min=None,
                      clip_max=None,
                      missing_value=None):
  # Clip min and max if desired.
  if clip_min is not None:
    features = np.maximum(features, clip_min)
    features = np.append(features, clip_min)
  if clip_max is not None:
    features = np.minimum(features, clip_max)
    features = np.append(features, clip_max)
  # Make features unique.
  unique_features = np.unique(features)
  # Remove missing values if specified.
  if missing_value is not None:
    unique_features = np.delete(unique_features,
                                np.where(unique_features == missing_value))

  # Compute and return quantiles over unique non-missing feature values.
  return np.quantile(
      unique_features,
      np.linspace(0., 1., num=num_keypoints),
      interpolation='nearest').astype(float)

def compute_uniform(dataset,column,num_keypoints):
    if(dataset=='cover'):
        _max = [3858,360, 66, 1397, 601, 7117, 254, 254, 254, 7173]
        _min = [1859, 0, 0, 0,-173, 0, 0, 0, 0, 0]
    if(dataset=='dmv'):
        _max = [20000,2020,8000]
        _min = [10000,1900,1]
    if(dataset=='TPCH'):
        _max = [50,2219,2217]
        _min = [1,0,0]
    points = np.linspace(_min[column], _max[column], num=num_keypoints)
    return points

def buildEqualDepthHist(data,nbin=100):
    bins = np.quantile(data, np.linspace(0,1,nbin+1))
    bin_size = len(data)/nbin
    return bins,bin_size

def computeRangeCardinality(bins,bin_size,left,right):
    if(left>=right):
        return False
    if(right<bins[0] or left>bins[-1]):
        return 0

    for l in range(len(bins)):
        if(bins[l]>=left):
            break
    for r in range(len(bins)-1,-1,-1):
        if(bins[r]<=right):
            break
    if(l==r and l==len(bins)-1):
        return(bins[l]-left)/(bins[l]-bins[l-1])*bin_size
    elif(l==r and l==0):
        return(right-bins[l])/(bins[l+1]-bins[l])*bin_size
    elif(l==r):
        return ((bins[l]-left)/(bins[l]-bins[l-1]) + (right-bins[r])/(bins[r+1]-bins[r]))*bin_size
    elif(l==0 and r==len(bins)-1):
        return bin_size * (len(bins)-1)
    elif(l==0):
        return (r-l + (right-bins[r])/(bins[r+1]-bins[r]))*bin_size
    elif(r==len(bins)-1):
        return (r-l + (bins[l]-left)/(bins[l]-bins[l-1]))*bin_size
    else:
        return (r-l + (bins[l]-left)/(bins[l]-bins[l-1]) + (right-bins[r])/(bins[r+1]-bins[r]))*bin_size


def q_error_loss(label, pred):
    label = tf.where(tf.equal(label, 0), tf.convert_to_tensor(1e-10), label)
    pred = tf.where(tf.equal(pred, 0), tf.convert_to_tensor(1e-10), pred)
    # label = tf.math.exp(label)
    # pred = tf.math.exp(pred)
    # label = tf.math.multiply(label,581012)
    # pred = tf.math.multiply(pred,581012)
    q1 = tf.math.divide(label,pred)
    q2 = tf.math.divide(pred,label)
    q = tf.math.maximum(q1,q2)
    return tf.math.subtract(tf.reduce_mean(q,axis=-1),1)   

def split_data(data,label,num_part):
    data = np.split(data,num_part,axis=0)
    label = np.split(label,num_part,axis=0)
    return data,label

def random_from_data(data,label,size):
    idx = np.random.choice(np.arange(len(data)), int(size), replace=False)
    data_sample = data[idx]
    label_sample = label[idx]
    return data_sample,label_sample

def cdf_range(lattice_model, feature,dimension):
    assert feature.shape[-1] == 2*dimension
    s = []
    for i in range(2*dimension):
        s.append(feature[:,i])
    # s把原来的N维特征拆开到list中

    x = []
    for i in range(2**dimension):
        tmp = []
        indicator = (dimension-len(bin(i).replace("0b","")))*'0'+(bin(i).replace("0b",""))
        for j in range(dimension):
            tmp.append(s[2*j+int(indicator[j])])
        x.append(tmp)
    # x是每个point的特征输入

    F = []
    for i in range(2**dimension):
        F.append(np.clip(lattice_model.predict(x[i]),0,1))

    pred = 0
    for i in range(2**dimension):
        square = 0
        for each in bin(i).replace("0b",""):
            square += int(each)
        pred += (-1)**(dimension-square)*F[i]

    return np.clip(pred,0,1)