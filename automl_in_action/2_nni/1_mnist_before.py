import argparse
import logging
import math
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# code from https://github.com/microsoft/nni/blob/master/examples/trials/mnist/mnist_before.py

FLAGS = None

logger = logging.getLogger('mnist_AutoML')

class MnistNetwork(object):
    def __init__(self,
                 channel_1_num,
                 channel_2_num,
                 conv_size,
                 hidden_size,
                 pool_size,
                 learning_rate,
                 x_dim=784,
                 y_dim=10):
        self.channel_1_num = channel_1_num
        self.channel_2_num = channel_2_num
        self.conv_size =