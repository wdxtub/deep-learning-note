#!/bin/bash

export MASTER=spark://wangdadeMacBook-Pro.local:7077
# absolute path
export tfos=/Users/dawang/Documents/GitHub/deep-learning-note/automl_in_action
rm -rf ${tfos}/data/mnist/csv
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
${tfos}/4_tfos/1_mnist_data_setup.py \
--output ${tfos}/data/mnist/csv \
--format csv