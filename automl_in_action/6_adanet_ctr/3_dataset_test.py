import adanet
from adanet.examples import simple_dnn
import tensorflow as tf
import os
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split


EPOCH = 10
BATCH_SIZE = 64
RANDOM_SEED = 42
NUM_CLASSES = 2

LOG_DIR = 'models'

tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.enable_eager_execution()


def df_to_dataset(dataframe, features, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe[features]), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    # ds = ds.batch(batch_size)
    return ds


print("读取数据")
data = pd.read_csv('./data/criteo_train_small.txt')
split_line = "==================================================="

DENSE_KEY = 'dense'

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

# 填充空值 Fill NA/NaN values using the specified method
# 稀疏的填写 -1，数值的填写 0
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )

# 设定 feature_columns
feature_columns = [tf.feature_column.numeric_column(name) for name in dense_features]

loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
head = tf.contrib.estimator.binary_classification_head(loss_reduction=loss_reduction)

train, test = train_test_split(data, test_size=0.2)

train_ds = df_to_dataset(train, dense_features, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test,dense_features, shuffle=False, batch_size=BATCH_SIZE)

print( feature_columns)

for feat, targ in test_ds.take(1):
    print ('Features: {}, Target: {}'.format(feat, targ))