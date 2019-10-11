import adanet
from adanet.examples import simple_dnn
import tensorflow as tf
import os
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

# 根据 Adanet 论文的参数进行设置

BATCH_SIZE = 100
RANDOM_SEED = 42
NUM_CLASSES = 2

LOG_DIR = 'models'
RESULT_DIR = 'results'

# tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

LR = 0.001
LS = 512
TRAIN_STEPS = 2000
ADANET_ITERATIONS = 30

'''
"""
数据对比 Linear 是 Baseline

CTR      | Linear | DNN1  
Acc(%)   | 75.62  | 73.66     
AUC      | 0.635  | 0.65       
time(s)  |   21   |  1257      


DNN1 = loss SUM_BY_NONZERO_WEIGHTS


耗时 1 = MBP 2018 15 寸 i7 2.2GHz

重要参数有 4 个
B 是层数 125, 256, 512
n 是学习率 0.0001 或 0.1


Train Step 均为 2000, SUM_OVER_NONE_ZERO_WEIGHTS
125 + 0.0001 + 30 轮 = 0.7306 / 0.587
512 + 0.1 + 20 轮 = 0.754625 / 0.50
512 + 0.0001 + 20 轮 = 0.7193 / 0.586
512 + 0.0001 + 30 轮 = 0.6437 / 0.55
512 + 0.001 + 20 轮 = 0.7403 / 0.579
256 + 0.001 + 20 轮 = 0.7529 / 0.5538

Train Step 均为 2000, SUM_OVER_BATCH_SIZE
512 + 0.1 + 20 轮 = 0.7465 / 0.597
512 + 0.0001 + 30 轮 = 0.5978 / 0.5637 

512 + 0.01 + 30 轮 = 0.7546 / 0.6498 (2019.09.20) Test 数据（需要与 Train 进行对比）
512 + 0.001 + 30 轮 = 0.7143 / 0.6165 

"""

'''

# 这里要使用 dict 来包住 features，不然会被合成一个向量，后续无法处理
def df_to_dataset(dataframe, features, shuffle=True):
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe[features]), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


print("读取数据")
data = pd.read_csv('./data/criteo_train_small.txt')
split_line = "==================================================="


DENSE_KEY = 'dense'
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

# 填充空值 Fill NA/NaN values using the specified method
# 稀疏的填写 -1，数值的填写 0
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )

# 设定 feature_columns
feature_columns = [tf.feature_column.numeric_column(name) for name in dense_features]

loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
# 注：这里用 binary 会提示 Trapezoidal rule is known to produce incorrect PR-AUCs
head = tf.contrib.estimator.binary_classification_head(loss_reduction=loss_reduction)


train, test = train_test_split(data, test_size=0.2)

# tf.enable_eager_execution()

def input_fn(mode):
    """Generate an input_fn for the Estimator."""
    def _input_fn():
        if mode == 'train':
            ds = df_to_dataset(train, dense_features)
        elif mode == 'valid':
            ds = df_to_dataset(train, dense_features, shuffle=False)
        elif mode == 'test':
            ds = df_to_dataset(test, dense_features, shuffle=False)

        ds = ds.batch(BATCH_SIZE)
        iterator = ds.make_one_shot_iterator()
        return iterator.get_next()

    return _input_fn


def time_str(now):
    return now.strftime("%Y%m%d_%H%M%S")


def linear_ada():

    print("==============================================")
    start = datetime.datetime.now()
    print("Start Train Adanet with [Linear Model] on Criteo at %s" % time_str(start))
    print("- - - - - - - - - - - - - - - - - - - - - - - -")

    LEARNING_RATE = 0.001
    TRAIN_STEPS = 5000

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
            input_fn=input_fn("train"),
            max_steps=TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=input_fn("test"),
            steps=None)
    )


    print("Accuracy:", results["accuracy"])
    print("AUC", results["auc"])
    print("Loss:", results["average_loss"])

    end = datetime.datetime.now()
    print("Training end at %s" % time_str(end))
    print("Time Spend %s" % str(end - start))

    print("==============================================")


def dnn_ada():
    print("==============================================")
    start = datetime.datetime.now()
    print("Start Train Adanet with [DNN Model] on Criteo at %s" % time_str(start))
    print("- - - - - - - - - - - - - - - - - - - - - - - -")

    # 根据论文参数调整
    LEARNING_RATE = LR

    model_dir = os.path.join(LOG_DIR, "dnn_%s" % time_str(start))
    result_file = os.path.join(RESULT_DIR, "dnn_%s" % time_str(start))
    valid_file = os.path.join(RESULT_DIR, "valid_%s" % time_str(start))
    test_file = os.path.join(RESULT_DIR, "test_%s" % time_str(start))
    tpred_file = os.path.join(RESULT_DIR, "tpred_%s" % time_str(start))
    vpred_file = os.path.join(RESULT_DIR, "vpred_%s" % time_str(start))

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=50000,
        save_summary_steps=50000,
        tf_random_seed=RANDOM_SEED,
        model_dir=model_dir
    )

    # layer size 125 256 512
    estimator = adanet.Estimator(
        head=head,
        subnetwork_generator=simple_dnn.Generator(
            feature_columns=feature_columns,
            layer_size=LS,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
            seed=RANDOM_SEED),
        max_iteration_steps=TRAIN_STEPS // ADANET_ITERATIONS,
        evaluator=adanet.Evaluator(
            input_fn=input_fn("train"),
            steps=None),
        config=config
    )

    results, _ = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=input_fn("train"),
            max_steps=TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=input_fn("test"),
            steps=None)
    )

    print("Accuracy:", results["accuracy"])
    print("AUC", results["auc"])
    print("Loss:", results["average_loss"])

    # 重新获取评测结果
    train_spec = estimator.evaluate(input_fn=input_fn("train"))
    test_spec = estimator.evaluate(input_fn=input_fn("test"))

    end = datetime.datetime.now()
    print("Training end at %s" % time_str(end))
    print("Time Spend %s" % str(end - start))
    print("==============================================")
    with open('{}.txt'.format(result_file), 'w') as f:
        f.write('Train Configs:\n')
        f.write('[Layer Size] {}\n'.format(LS))
        f.write('[Learning Rate] {}\n'.format(LR))
        f.write('[BATCH SIZE] {}\n'.format(BATCH_SIZE))
        f.write('[Train Step] {}\n'.format(TRAIN_STEPS))
        f.write('[Adanet Iteration] {}\n'.format(ADANET_ITERATIONS))
        f.write('\nResults:\n')
        f.write('[Accurary] {}\n'.format(results["accuracy"]))
        f.write('[AUC] {}\n'.format(results["auc"]))
        f.write('[Loss] {}\n'.format(results["average_loss"]))
        f.write('[Time Spend] {}\n'.format(str(end - start)))
        f.write('[Train Spec] {}\n'.format(str(train_spec)))
        f.write('[Test Spec] {}\n'.format(str(test_spec)))

    # 写入测试集
    print("export test data")
    test.to_csv('{}.txt'.format(test_file))
    print("export train data")
    train.to_csv('{}.txt'.format(valid_file))

    # 进行预测
    predictions = estimator.predict(input_fn=input_fn("test"))
    # 写入预测集
    with open('{}.txt'.format(tpred_file), 'w') as f:
        for pred in predictions:
            f.write(str(pred))
            f.write('\n')

    # 进行预测并写入预测集
    predictions = estimator.predict(input_fn=input_fn("valid"))
    # 写入预测集
    with open('{}.txt'.format(vpred_file), 'w') as f:
        for pred in predictions:
            f.write(str(pred))
            f.write('\n')



def ensemble_ada():
    print("==============================================")
    start = datetime.datetime.now()
    print("Start Train Adanet with [Ensemble Model] on Criteo at %s" % time_str(start))
    print("- - - - - - - - - - - - - - - - - - - - - - - -")

    LEARNING_RATE = 0.003
    TRAIN_STEPS = 5000
    ADANET_ITERATIONS = 10

    model_dir = os.path.join(LOG_DIR, "dnn_%s" % time_str(start))

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=50000,
        save_summary_steps=50000,
        tf_random_seed=RANDOM_SEED,
        model_dir=model_dir
    )

    estimator = adanet.AutoEnsembleEstimator(
        head=head,
        candidate_pool={
            "linear":
                tf.estimator.LinearEstimator(
                    head=head,
                    feature_columns=feature_columns,
                    optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
                    config=config
                ),
            "dnn":
                tf.estimator.DNNEstimator(
                     head=head,
                     feature_columns=feature_columns,
                     optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
                     config=config,
                     hidden_units=[200, 150, 100]
                 )
        },
        max_iteration_steps=5
    )

    results, _ = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=input_fn("train"),
            max_steps=TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=input_fn("test"),
            steps=None)
    )

    print("Accuracy:", results["accuracy"])
    print("AUC", results["auc"])
    print("Loss:", results["average_loss"])

    end = datetime.datetime.now()
    print("Training end at %s" % time_str(end))
    print("Time Spend %s" % str(end - start))
    print("==============================================")
    # 写入到结果文件



if __name__ == "__main__":
    # linear_ada()
    dnn_ada()
    # ensemble_ada()
