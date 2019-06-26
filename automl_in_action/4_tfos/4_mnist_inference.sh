#!/bin/bash

# /Users/dawang/Documents/GitHub/deep-learning-note/automl_in_action/4_tfos
export MASTER=spark://wangdadeMacBook-Pro.local:7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export tfos=/Users/dawang/Documents/GitHub/deep-learning-note/automl_in_action
export output=models/mnist
export prediction=models/mnist_prediction

rm -rf ${tfos}/4_tfos/${prediction}

${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files ${tfos}/4_tfos/mnist_dist.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${tfos}/4_tfos/2_mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images ${tfos}/data/mnist/csv/test/images \
--labels ${tfos}/data/mnist/csv/test/labels \
--format csv \
--mode inference \
--model ${output} \
--output ${prediction}

echo "Model Predictions"
less ${prediction}/part-00000