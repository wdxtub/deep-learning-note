import sys
import os
from sklearn import metrics

# 结果文件存放目录
RSDIR = 'results'
# 模型存放路径
MDDIR = 'models'


# check result/model files
def check_files(datestr):
    prefixlist = ['vpred', 'tpred', 'valid', 'test', 'dnn']
    for prefix in prefixlist:
        filepath = os.path.join(RSDIR, f'{prefix}_{datestr}.txt')
        if not os.path.exists(filepath):
            print(f'file {filepath} not exists! please check the datestr parameter.')
            return False
    filepath = os.path.join(MDDIR, f'dnn_{datestr}')
    if not os.path.exists(filepath):
        print(f'model dir {filepath} not exists! please check the datestr parameter.')
        return False
    return True


# parse result from adanet
def parse_result(datestr):
    y_test_true = []
    y_test_pred = []
    y_valid_true = []
    y_valid_pred = []
    test_path = os.path.join(RSDIR, f'test_{datestr}.txt')
    tpred_path = os.path.join(RSDIR, f'tpred_{datestr}.txt')
    valid_path = os.path.join(RSDIR, f'valid_{datestr}.txt')
    vpred_path = os.path.join(RSDIR, f'vpred_{datestr}.txt')
    # 处理测试数据
    with open(test_path, 'r') as f:
        for line in f:
            arr = line.split(',')
            # 排除其他的结果
            if not (arr[1] == '1' or arr[1] == '0'):
                continue
            # 记录下来
            y_test_true.append(int(arr[1]))
    with open(valid_path, 'r') as f:
        for line in f:
            arr = line.split(',')
            # 排除其他的结果
            if not (arr[1] == '1' or arr[1] == '0'):
                continue
            # 记录下来
            y_valid_true.append(int(arr[1]))
    with open(tpred_path, 'r') as f:
        for line in f:
            idx = line.find('probabilities')
            start = line.find('[', idx)
            end = line.find(']', start)
            tmp = line[start+1:end].replace(' ', '')
            arr = tmp.split(',')
            # 记录第二个
            y_test_pred.append(float(arr[1]))
    with open(vpred_path, 'r') as f:
        for line in f:
            idx = line.find('probabilities')
            start = line.find('[', idx)
            end = line.find(']', start)
            tmp = line[start+1:end].replace(' ', '')
            arr = tmp.split(',')
            # 记录第二个
            y_valid_pred.append(float(arr[1]))

    # 计算测试的 auc
    test_auc = metrics.roc_auc_score(y_test_true, y_test_pred)
    print(f'[test auc] {test_auc}')
    valid_auc = metrics.roc_auc_score(y_valid_true, y_valid_pred)
    print(f'[valid auc] {valid_auc}')


# 这个文件要放在结果文件的上一层目录，并且结果文件必须放在 results 文件夹里
def show_help():
    print("Usage: python parse_result.py [datestr]")
    print("datestr sample: 20190923_141558")
    print("Notice: this script should be placed on the parent level of all the result files.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong Parameter Count!")
        show_help()
        sys.exit(-1)
    # 正式开始处理
    datestr = sys.argv[1]
    print(f'[dateStr] {datestr}')
    # 1. 检查文件
    if not check_files(datestr):
        sys.exit(-1)
    # 2. 解析结果
    parse_result(datestr)
