import sys
import os
from sklearn import metrics

VERSION = '20191127-192523'
TRAIN_SOURCE = '/Users/dawang/Desktop/tfos_test/data/part-00001'
TEST_SOURCE = '/Users/dawang/Desktop/tfos_test/test/part-00002'
TRAIN_PRED = '/Users/dawang/Desktop/tfos_test/predictions/{}/train'.format(VERSION)
TEST_PRED = '/Users/dawang/Desktop/tfos_test/predictions/{}/test'.format(VERSION)

y_train_true = []
y_train_pred = []
y_test_true = []
y_test_pred = []

with open(TRAIN_SOURCE, 'r') as f:
    for line in f:
        arr = line.split(' ')
        if arr[0] == '0':
            y_train_true.append(0)
        else:
            y_train_true.append(1)

with open(TEST_SOURCE, 'r') as f:
    for line in f:
        arr = line.split(' ')
        if arr[0] == '0':
            y_test_true.append(0)
        else:
            y_test_true.append(1)

with open(TRAIN_PRED, 'r') as f:
    for line in f:
        idx = line.find('probabilities')
        start = line.find('[', idx)
        end = line.find(']', start)
        tmp = line[start+1:end].replace(' ', '')
        arr = tmp.split(',')
        # 记录第二个
        y_train_pred.append(float(arr[1]))

with open(TEST_PRED, 'r') as f:
    for line in f:
        idx = line.find('probabilities')
        start = line.find('[', idx)
        end = line.find(']', start)
        tmp = line[start+1:end].replace(' ', '')
        arr = tmp.split(',')
        # 记录第二个
        y_test_pred.append(float(arr[1]))

print('train pred/true length', len(y_train_pred), len(y_train_true))
print('test pred/true length', len(y_test_pred), len(y_test_true))

test_auc = metrics.roc_auc_score(y_test_true, y_test_pred)
print(f'[test auc] {test_auc}')
train_auc = metrics.roc_auc_score(y_train_true, y_train_pred)
print(f'[train_auc] {train_auc}')