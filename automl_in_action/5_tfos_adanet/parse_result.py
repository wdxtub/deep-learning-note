import tensorflow as tf
from pathlib import Path

import datetime
import random
import numpy
import sys

print(sys.argv)

version = '/Users/dawang/Desktop/tfos_test/models/1575611737'
testfile = '/Users/dawang/Desktop/tfos_test/test/part-00002'
print('version', version)
print('testfile', testfile)

pb_file_path = version

# 输入的参数
input_dim = 500


def build_example(line):
    parts = line.split(' ')
    label = int(parts[0])
    if label > 1:
        label = 1

    indice_list = []
    items = parts[1:]
    for item in items:
        index = int(item.split(':')[0])
        if index >= input_dim:
            continue
        indice_list += [[0, index]]

    value_list = [1 for i in range(len(indice_list))]
    shape_list = [1, input_dim]

    indice_list = numpy.asarray(indice_list)
    value_list = numpy.asarray(value_list)
    shape_list = numpy.asarray(shape_list)
    return indice_list, value_list, shape_list, label


# 一定要放在 with 里，不然 导出的 graph 不带变量和参数
with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], pb_file_path)
    signature = meta_graph_def.signature_def
    # print(signature)

    signature_key = "predict"
    y_tensor_name = signature[signature_key].outputs["probabilities"].name
    y = sess.graph.get_tensor_by_name(y_tensor_name)
    print('y', y_tensor_name)
    indices = signature[signature_key].inputs['indices'].name
    indice_tensor = sess.graph.get_tensor_by_name(indices)
    print('indices', indices)
    values = signature[signature_key].inputs["values"].name
    value_tensor = sess.graph.get_tensor_by_name(values)
    print('values', values)
    shape = signature[signature_key].inputs["dense_shape"].name
    shape_tensor = sess.graph.get_tensor_by_name(shape)
    print('shape', shape)

    # 每行读取 testfile
    one_count = 0
    zero_count = 0
    pone_count = 0
    count = 0
    with open('{}.result'.format(version), 'w') as w:
        with open(testfile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                indice_list, value_list, shape_list, label = build_example(line)
                if len(indice_list) == 0:
                    continue
                if label == 1:
                    one_count = one_count + 1
                else:
                    zero_count = zero_count + 1

                if zero_count % 10000 == 0:
                    print('working....', zero_count)

                # 这样就可以进行预测
                y_out = sess.run(y[:, 0], feed_dict={
                    indice_tensor: indice_list,
                    value_tensor: value_list,
                    shape_tensor: shape_list})
                count = count + 1
                print(count)
                if y_out[0] < 0.5:
                    pone_count = pone_count + 1
                w.write('{} {}\n'.format(label, y_out[0]))
    print('#one', one_count)
    print('#zero', zero_count)
    print('#pone', pone_count)



# test_auc = metrics.roc_auc_score(y_test_true, y_test_pred)
# print(f'[test auc] {test_auc}')
# train_auc = metrics.roc_auc_score(y_train_true, y_train_pred)
# print(f'[train_auc] {train_auc}')