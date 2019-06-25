import functools

import adanet
from adanet.examples import simple_dnn
import tensorflow as tf
import os
import datetime

# Fix Random Seed
RANDOM_SEED = 42

tf.logging.set_verbosity(tf.logging.ERROR)

LOG_DIR = 'models'

# 可选数据集
# tf.keras.datasets.mnist
# tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = (tf.keras.datasets.fashion_mnist.load_data())

FEATURES_KEY = "images"

NUM_CLASSES = 10

loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

head = tf.contrib.estimator.multi_class_head(NUM_CLASSES, loss_reduction=loss_reduction)

feature_columns = [
    tf.feature_column.numeric_column(FEATURES_KEY, shape=[28, 28, 1])
]


"""
数据对比

MNIST    | Linear | DNN  | CNN
准确率(%) | 92.48  | 95.98 | 98.72
耗时1(s)  | 32    | 55    | 74
耗时2(s)  | 71    | 117   | 143

FASHINON | Linear | DNN  | CNN
准确率(%) | 92.48  | 95.98 | 98.72
耗时1(s)  | 32    | 55    | 74
耗时2(s)  | 71    | 117   | 143


耗时 1 = MBP 2018 15 寸 i7 2.2GHz
耗时 2 = DELL 7490 i5 8250U 1.6Ghz
"""

# MBP 2018: 32s
    # DELL 7490 WSL: 1m11s

# Accuracy: 0.9598
# MBP 2018: 55s
# DELL 7490 WSL: 1m57s

# Accuracy: 0.9872
    # MBP 2018: 1m14s
    # DELL 7490 WSL: 2m23s

def generator(images, labels):
    """Returns a generator that returns image-label pairs."""
    def _gen():
        for image, label in zip(images, labels):
            yield image, label
    return _gen


def preprocess_image(image, label):
    """Preprocesses an image for an `Estimator`."""
    image = image / 255.
    image = tf.reshape(image, [28, 28, 1])
    features = {FEATURES_KEY: image}
    return features, label


def input_fn(partition, training, batch_size):
    """Generate an input_fn for the Estimator."""

    def _input_fn():
        if partition == "train":
            dataset = tf.data.Dataset.from_generator(
                generator(x_train, y_train), (tf.float32, tf.int32), ((28, 28), ()))
        else:
            dataset = tf.data.Dataset.from_generator(
                generator(x_test, y_test), (tf.float32, tf.int32), ((28, 28), ()))

        if training:
            dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat()

        dataset = dataset.map(preprocess_image).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
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
    BATCH_SIZE = 64

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
            input_fn=input_fn("train", training=True, batch_size=BATCH_SIZE),
            max_steps=TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=input_fn("test", training=False, batch_size=BATCH_SIZE),
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
    BATCH_SIZE = 64
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
            input_fn=input_fn("train", training=False, batch_size=BATCH_SIZE),
            steps=None),
        config=config
    )

    results, _ = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=input_fn("train", training=True, batch_size=BATCH_SIZE),
            max_steps=TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=input_fn("test", training=False, batch_size=BATCH_SIZE),
            steps=None)
    )

    print("Accuracy:", results["accuracy"])
    print("Loss:", results["average_loss"])

    end = datetime.datetime.now()
    print("Training end at %s" % time_str(end))
    print("Time Spend %s" % str(end - start))
    print("==============================================")


class SimpleCNNBuilder(adanet.subnetwork.Builder):
    """Builds a CNN subnetwork for AdaNet."""

    def __init__(self, learning_rate, max_iteration_steps, seed):
        """Initializes a `SimpleCNNBuilder`.

        Args:
          learning_rate: The float learning rate to use.
          max_iteration_steps: The number of steps per iteration.
          seed: The random seed.

        Returns:
          An instance of `SimpleCNNBuilder`.
        """
        self._learning_rate = learning_rate
        self._max_iteration_steps = max_iteration_steps
        self._seed = seed

    def build_subnetwork(self,
                       features,
                       labels,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""
        images = tf.to_float(features[FEATURES_KEY])
        kernel_initializer = tf.keras.initializers.he_normal(seed=self._seed)
        x = tf.layers.conv2d(
            images,
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=kernel_initializer)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(
            x, units=64, activation="relu", kernel_initializer=kernel_initializer)

        # The `Head` passed to adanet.Estimator will apply the softmax activation.
        logits = tf.layers.dense(
            x, units=10, activation=None, kernel_initializer=kernel_initializer)

        # Use a constant complexity measure, since all subnetworks have the same
        # architecture and hyperparameters.
        complexity = tf.constant(1)

        return adanet.Subnetwork(
            last_layer=x,
            logits=logits,
            complexity=complexity,
            persisted_tensors={})

    def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
        """See `adanet.subnetwork.Builder`."""

        # Momentum optimizer with cosine learning rate decay works well with CNNs.
        learning_rate = tf.train.cosine_decay(
            learning_rate=self._learning_rate,
            global_step=iteration_step,
            decay_steps=self._max_iteration_steps)
        optimizer = tf.train.MomentumOptimizer(learning_rate, .9)
        # NOTE: The `adanet.Estimator` increments the global step.
        return optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                       iteration_step, summary):
        """See `adanet.subnetwork.Builder`."""
        return tf.no_op("mixture_weights_train_op")

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""
        return "simple_cnn"


class SimpleCNNGenerator(adanet.subnetwork.Generator):
    """Generates a `SimpleCNN` at each iteration. """
    def __init__(self, learning_rate, max_iteration_steps, seed=None):
        """Initializes a `Generator` that builds `SimpleCNNs`.

        Args:
          learning_rate: The float learning rate to use.
          max_iteration_steps: The number of steps per iteration.
          seed: The random seed.

        Returns:
          An instance of `Generator`.
        """
        self._seed = seed
        self._cnn_builder_fn = functools.partial(
            SimpleCNNBuilder,
            learning_rate=learning_rate,
            max_iteration_steps=max_iteration_steps)

    def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
        """See `adanet.subnetwork.Generator`."""
        seed = self._seed
        # Change the seed according to the iteration so that each subnetwork
        # learns something different.
        if seed is not None:
            seed += iteration_number
        return [self._cnn_builder_fn(seed=seed)]


def cnn_ada():
    print("==============================================")
    start = datetime.datetime.now()
    print("Start Train Adanet with [CNN Model] on Mnist at %s" % time_str(start))
    print("- - - - - - - - - - - - - - - - - - - - - - - -")

    LEARNING_RATE = 0.05  # @param {type:"number"}
    TRAIN_STEPS = 5000  # @param {type:"integer"}
    BATCH_SIZE = 64  # @param {type:"integer"}
    ADANET_ITERATIONS = 2  # @param {type:"integer"}

    model_dir = os.path.join(LOG_DIR, "cnn_%s" % time_str(start))

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=50000,
        save_summary_steps=50000,
        tf_random_seed=RANDOM_SEED,
        model_dir=model_dir
    )

    max_iteration_steps = TRAIN_STEPS // ADANET_ITERATIONS
    estimator = adanet.Estimator(
        head=head,
        subnetwork_generator=SimpleCNNGenerator(
            learning_rate=LEARNING_RATE,
            max_iteration_steps=max_iteration_steps,
            seed=RANDOM_SEED),
        max_iteration_steps=max_iteration_steps,
        evaluator=adanet.Evaluator(
            input_fn=input_fn("train", training=False, batch_size=BATCH_SIZE),
            steps=None),
        report_materializer=adanet.ReportMaterializer(
            input_fn=input_fn("train", training=False, batch_size=BATCH_SIZE),
            steps=None),
        adanet_loss_decay=.99,
        config=config
    )

    results, _ = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=input_fn("train", training=True, batch_size=BATCH_SIZE),
            max_steps=TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=input_fn("test", training=False, batch_size=BATCH_SIZE),
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
    cnn_ada()
