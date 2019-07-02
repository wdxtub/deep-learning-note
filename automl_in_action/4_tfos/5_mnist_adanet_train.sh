#!/bin/bash

# /Users/dawang/Documents/GitHub/deep-learning-note/automl_in_action/4_tfos
export MASTER=spark://wangdadeMacBook-Pro.local:7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export tfos=/Users/dawang/Documents/GitHub/deep-learning-note/automl_in_action
export output=models/mnist

rm -rf ${tfos}/4_tfos/${output}

${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files ${tfos}/4_tfos/mnist_dist.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${tfos}/4_tfos/3_mnist_spark_adanet.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--mode train \
--model ${output}

echo "Model Files"
ls -l ${output}