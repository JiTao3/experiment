import sys
import os
import datetime
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.abspath(os.getcwd()))
from utils.cdf_range import cdf_range_3d
from utils.data_utils import bound_result


def clone(model, N=3):
    return [tf.keras.models.clone_model(model) for _ in range(N)]


def train_multi_models(
    models,
    input_features,
    validation_featurees,
    target,
    validation_target,
    # loss,
    epochs,
    batch_size,
    learning_rate,
    # callback,
):
    for model in models:
        model.compile(
            loss=tf.keras.losses.mean_absolute_error,
            # loss=q_error_loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate, epsilon=1e-3),
        )
    count_model = 0
    for model in models:
        print("training NO.{} in models: ".format(count_model))
        # callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
        count_model += 1
        train_starttime = datetime.datetime.now()
        model.fit(
            input_features,
            target,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_featurees, validation_target),
            # callbacks=callback,
        )
        train_endtime = datetime.datetime.now()
    print("training time: ", (train_endtime - train_starttime))
    return models


def multi_model_predict(models, range_source):
    results = []
    for model in models:
        result = cdf_range_3d(model, range_source)
        result = bound_result(result, 0.0, 1.0)
        results.append(result)
    return results


def difference_result(results):
    diff = [
        max(results[0][i], results[1][i], results[2][i]) - min(results[0][i], results[1][i], results[2][i])
        for i in range(results[0].shape[0])
    ]
    return diff

