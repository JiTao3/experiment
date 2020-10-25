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
    print("Median: {}\n".format(np.median(qerror)))
    print("90th percentile: {}\n".format(np.percentile(qerror, 90)))
    print("95th percentile: {}\n".format(np.percentile(qerror, 95)))
    print("99th percentile: {}\n".format(np.percentile(qerror, 99)))
    print("Max: {}\n".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


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

def q_error_loss(label, pred):
    label = tf.where(tf.equal(label, 0), tf.convert_to_tensor(1e-10), label)
    pred = tf.where(tf.equal(pred, 0), tf.convert_to_tensor(1e-10), pred)
    label = tf.math.exp(label)
    pred = tf.math.exp(pred)
    # label = tf.math.multiply(label,581012)
    # pred = tf.math.multiply(pred,581012)
    q1 = tf.math.divide(label,pred)
    q2 = tf.math.divide(pred,label)
    q = tf.math.maximum(q1,q2)
    return tf.math.subtract(tf.reduce_mean(q,axis=-1),1)   
    