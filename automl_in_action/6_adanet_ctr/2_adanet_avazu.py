import adanet
from adanet.examples import simple_dnn
import tensorflow as tf
import os
import datetime
import pandas as pd

print('读取数据')
data = pd.read_csv('data/avazu_train_small')
split_line = "==================================================="

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

# 设定 feature_columns
feature_columns = [tf.feature_column.numeric_column('I'+str(i)) for i in range(1, 14)]

loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
head = tf.contrib.estimator.binary_classification_head(loss_reduction=loss_reduction)

EPOCH = 10
BATCH_SIZE = 64
RANDOM_SEED = 42
NUM_CLASSES = 2

LOG_DIR = 'models'

tf.logging.set_verbosity(tf.logging.ERROR)


# 全是 Sparse
# id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14-21

# label,I1-13,C1-26
def generator(ln):
    splits = tf.string_split([ln], delimiter=',')
    label = splits.values[0]
    # 解析 dense 部分
    features = {}
    for i in range(1, 14):
        features['I'+str(i)] = tf.string_to_number(splits.values[i], tf.int64)

    return features, label


def input_fn(partition):
    """Generate an input_fn for the Estimator."""
    def _input_fn():
        ds = tf.data.Dataset.list_files('data/criteo_train_small.txt').repeat(EPOCH)
        ds = ds.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=10))
        parse_fn = generator
        if partition == "train":
            ds = ds.map(parse_fn, num_parallel_calls=5).shuffle(BATCH_SIZE * 5)
        else:
            ds = ds.map(parse_fn, num_parallel_calls=5)

        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE)).prefetch(100)
        iterator = ds.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn


def time_str(now):
    return now.strftime("%Y%m%d_%H%M%S")


def linear_ada():

    print("==============================================")
    start = datetime.datetime.now()
    print("Start Train Adanet with [Linear Model] on Mnist at %s" % time_str(start))
    print("- - - - - - - - - - - - - - - - - - - - - - - -")

    LEARNING_RATE = 0.001
    TRAIN_STEPS = 5000

    model_dir = os.path.join(LOG_DIR, "linear_%s" % time_str(start))

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=50000,
        save_summary_steps=50000,
        tf_random_seed=RANDOM_SEED,
        model_dir=model_dir
    )

    # 先测试下线性模型
    estimator = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=NUM_CLASSES,
        optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
        loss_reduction=loss_reduction,
        config=config
    )

    results, _ = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=input_fn("train"),
            max_steps=TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=input_fn("test"),
            steps=None)
    )

    print("Accuracy:", results["accuracy"])
    print("Loss:", results["average_loss"])

    end = datetime.datetime.now()
    print("Training end at %s" % time_str(end))
    print("Time Spend %s" % str(end - start))

    print("==============================================")


def dnn_ada():
    print("==============================================")
    start = datetime.datetime.now()
    print("Start Train Adanet with [DNN Model] on Mnist at %s" % time_str(start))
    print("- - - - - - - - - - - - - - - - - - - - - - - -")

    LEARNING_RATE = 0.003
    TRAIN_STEPS = 5000
    ADANET_ITERATIONS = 2

    model_dir = os.path.join(LOG_DIR, "dnn_%s" % time_str(start))

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=50000,
        save_summary_steps=50000,
        tf_random_seed=RANDOM_SEED,
        model_dir=model_dir
    )

    estimator = adanet.Estimator(
        head=head,
        subnetwork_generator=simple_dnn.Generator(
            feature_columns=feature_columns,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
            seed=RANDOM_SEED),
        max_iteration_steps=TRAIN_STEPS // ADANET_ITERATIONS,
        evaluator=adanet.Evaluator(
            input_fn=input_fn("train"),
            steps=None),
        config=config
    )

    results, _ = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=input_fn("train"),
            max_steps=TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=input_fn("test"),
            steps=None)
    )

    print("Accuracy:", results["accuracy"])
    print("Loss:", results["average_loss"])

    end = datetime.datetime.now()
    print("Training end at %s" % time_str(end))
    print("Time Spend %s" % str(end - start))
    print("==============================================")


if __name__ == "__main__":
    linear_ada()
    dnn_ada()
