#! /bin/bash
line="========================================"
line1="- - - - - - - - - - - - -"
echo $line
echo "  Adanet + TFOS算法 Tensorflow On Spark 本地运行脚本"
echo "python 3.6.8, tensorflow 1.14.0, spark standalone 2.4"
echo $line1

export QUEUE=default
export PYTHON_ROOT=/Users/dawang/virtualenv/ctr36
export PYSPARK_PYTHON=${PYTHON_ROOT}/bin/python
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export MASTER=spark://wangdadeMacBook-Pro.local:7077

date_str=`date +'%Y%m%d-%H%M%S'`

echo "本次模型版本" ${date_str}

log_dir=/Users/dawang/Desktop/tfos_test/logs/${date_str}
export_dir=/Users/dawang/Desktop/tfos_test/models/${date_str}
prediction_dir=/Users/dawang/Desktop/tfos_test/predictions/${date_str}
# 这里的数据文件需要命名为 part-* 才能够被读取
data_dir=/Users/dawang/Desktop/tfos_test/data/
tfrecord_dir=/Users/dawang/Desktop/tfos_test/tfrecord/
script_dir=./scripts/train/

echo "创建 Trace 文件夹"
mkdir -p ${log_dir}/trace

echo "Path Config"
echo "[PYTHON_ROOT]" ${PYTHON_ROOT}
echo "[PYSPARK_PYTHON]" ${PYSPARK_PYTHON}
echo "[Scirpts Dir]" ${script_dir}
echo "[Log Dir]" ${log_dir}
echo "[Export Dir]" ${export_dir}
echo "[Prediction Dir]" ${prediction_dir}
echo "[Data Dir]" ${data_dir}
echo $line

# echo "清理 ${export_dir}"
# rm -r ${export_dir}/*

#for mode in train inference export
for mode in train
do
    echo $line
    echo "${mode} Mode Start! `date +'%Y-%m-%d %H:%M:%S'`"
    echo $line1
	spark-submit \
	--master ${MASTER} \
	--deploy-mode client \
	--queue ${QUEUE} \
	--num-executors 3 \
	--executor-memory 2G \
	--py-files ${script_dir}tfspark.zip,${script_dir}adactr_dist.py \
	--conf spark.dynamicAllocation.enabled=false \
	--conf spark.yarn.maxAppAttempts=1 \
	--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_CUDA:$LIB_JVM:$LIB_HDFS \
	${script_dir}adactr.py \
	--format libsvm \
	--mode ${mode} \
	--log_dir ${log_dir} \
	--export_dir ${export_dir} \
	--data_dir ${data_dir} \
	--tfrecord_dir ${tfrecord_dir} \
	--cluster_size 3 \
	--prediction_dir ${prediction_dir} \
	--save_checkpoint_steps 5000 \
	--batch_size 128
done
