from pyspark.context import SparkContext
from pyspark.conf import SparkConf

import argparse
import numpy
import tensorflow as tf
from datetime import datetime

from tensorflowonspark import TFCluster
import adactr_dist

sc = SparkContext(conf=SparkConf().setAppName("adactr_tfos"))
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

parser = argparse.ArgumentParser()
parser.add_argument("--format", help="example format: (csv|tfr)", choices=["csv", "tfr", "libsvm"], default="libsvm")
parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("--readers", help="number of reader/enqueue threads", type=int, default=1)
parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("--mode", help="train|inference", default="train")
parser.add_argument("--rdma", help="use rdma connection", default=False)
parser.add_argument("--shuffle_size", help="size of shuffle buffer", type=int, default=50000)
# Folder
parser.add_argument("--tfrecord_dir", help="HDFS path to DSP data for saving in tfrecord format")
parser.add_argument("--data_dir", help="HDFS path to DSP data in parallelized format")
parser.add_argument("--log_dir", help="HDFS path to save/load model during train/inference", default="mnist_model")
parser.add_argument("--export_dir", help="HDFS path to save/load model during train/inference", default="mnist_model")
parser.add_argument("--prediction_dir", help="HDFS path to save test/inference output", default="predictions")
# Hyperparameters
parser.add_argument("--input_dim", help="maximum input dimensions", type=int, default=500)
# Training
parser.add_argument("--epochs", help="number of epochs", type=int, default=10)
parser.add_argument("--batch_size", help="number of records per batch", type=int, default=3)
parser.add_argument("--steps", help="maximum number of steps", type=int, default=1e15)
parser.add_argument("--validate_step", help="which step to validate", type=int, default=0)
parser.add_argument("--learning_rate", help="learning_rate", type=float, default=0.0001)
parser.add_argument("--save_checkpoint_steps", help="save checkpoint steps", type=int, default=100)

args = parser.parse_args()
print("args:", args)

print("{0} ===== Start".format(datetime.now().isoformat()))

if args.mode == 'train':
    cluster = TFCluster.run(sc, adactr_dist.map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.TENSORFLOW)
    cluster.shutdown()
elif args.mode == 'inference':
    print('inference part')
elif args.mode == 'export':
    print('export part')
elif args.mode == 'prepare':
    print('Convert Libsvm Data to TFrecord')

print("{0} ===== Stop".format(datetime.now().isoformat()))