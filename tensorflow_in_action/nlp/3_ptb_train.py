import numpy as np
import tensorflow as tf

TRAIN_DATA = "data/ptb.train"
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

# 读取文件
def read_data(file_path):
    with open(file_path, "r") as fin:
        # 将整个文档读进一个长字符串
        id_string = ' '.joint([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list

def make_batches(id_list, batch_size, num_step):
    # 计算总的额 batch 数量，每个 batch 包含的单词数量是 batch_size * num_step
    num_batches = (len(id_list) - 1) // (batch_size * num_step)
    # 整理成维度为 [batch_size, num_batches * num_step] 的二维数组
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # 沿着第二个维度将数据切分成 num_batches 个 batch，存入一个数组
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述操作，但每个位置向右移动一位。这里得到的是 RNN 每一步输出所需要预测的下一个单词
    label = np.array(id_list[1 : num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)
    # 返回一个长度为 num_batches 的数组，其中每一项包括一个 data 矩阵和一个 label 矩阵
    return list(zip(data_batches, label_batches))


def main():
    train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    # 这里插入模型训练的代码


if __name__ == "__main__":
    main()