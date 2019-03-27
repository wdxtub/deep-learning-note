import tensorflow as tf

# 定义网络结构
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 通过 tf.get_variable 来获取变量
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weigths", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 如果给出了正则生成函数，加入 losses 集合
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义前向传播
def inference(input_tensor, regularizer):
    # 声明第一层神经网络
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 声明第二层
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
