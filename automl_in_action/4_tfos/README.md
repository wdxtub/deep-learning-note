# Tensorflow On Spark

Reference: https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_Standalone

## 启动本地 Spark 集群

```bash
export MASTER=spark://wangdadeMacBook-Pro.local:7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=1 
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES})) 
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}
```

可以直接使用 `./1_start_spark.sh` 启动，启动后可以用 `jps` 命令查看或直接访问 `http://127.0.0.1:8080`，看看有没有两个 Worker

## 测试 Python 环境

直接输入 `pyspark` 看看有没有出来交互式窗口，如果有出现，在里面输入

```bash
>>> import tensorflow as tf
>>> from tensorflowonspark import TFCluster
>>> exit()
```

如果没有报错，可以继续下一步

## 用 Spark 转换 MNIST 数据

```bash
# /Users/dawang/Documents/GitHub/deep-learning-note/automl_in_action/4_tfos
export MASTER=spark://wangdadeMacBook-Pro.local:7077
export tfos=/Users/dawang/Documents/GitHub/deep-learning-note/automl_in_action
rm -rf ${tfos}/data/mnist/csv
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
{tfos}/4_tfos/1_mnist_data_setup.py \
--output ${tfos}/data/mnist/csv \
--format csv
```

可以直接用 `./2_mnist_preprocess.sh`

## 用 Spark 来进行训练

```bash
# /Users/dawang/Documents/GitHub/deep-learning-note/automl_in_action/4_tfos
export MASTER=spark://wangdadeMacBook-Pro.local:7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=1 
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export tfos=/Users/dawang/Documents/GitHub/deep-learning-note/automl_in_action
rm -rf ${tfos}/4_tfos/mnist_model
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files ${tfos}/4_tfos/mnist_dist.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${tfos}/4_tfos/mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images ${tfos}/data/mnist/csv/train/images \
--labels ${tfos}/data/mnist/csv/train/labels \
--format csv \
--mode train \
--model mnist_model
```