import functools

import adanet
from adanet.examples import simple_dnn
import tensorflow as tf
import os

# Fix Random Seed
RANDOM_SEED = 42

LOG_DIR = 'models'

(x_train, y_train), (x_test, y_test) = (tf.keras.datasets.mnist.load_data())

FEATURES_KEY = "images"


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


NUM_CLASSES = 10

loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

head = tf.contrib.estimator.multi_class_head(NUM_CLASSES, loss_reduction=loss_reduction)

feature_columns = [
    tf.feature_column.numeric_column(FEATURES_KEY, shape=[28, 28, 1])
]

LEARNING_RATE = 0.001
TRAIN_STEPS = 5000
BATCH_SIZE = 64

model_dir = os.path.join(LOG_DIR, "linear")

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