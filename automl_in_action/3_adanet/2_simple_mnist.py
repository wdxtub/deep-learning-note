import functools

import adanet
from adanet.examples import simple_dnn
import tensorflow as tf

# Fix Random Seed
RANDOM_SEED = 42

(x_train, y_train), (x_test, y_test) = (tf.keras.datasets.mnist.load_data())

FEATURES_KEY = "images"


def generator(images, labels):
    """Returns a generator that returns image-label pairs."""
    def _gen():
        for image, label in zip(images, labels):
            yield image, label
    return _gen

