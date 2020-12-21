from typing import Tuple
from matplotlib.pyplot import axes, fignum_exists, xlabel
from numpy.core.defchararray import title
import pandas as pd
import numpy as np

import matplotlib
import statsmodels
from statsmodels import test
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# 导入数据
train_pd = pd.read_csv("./data/give_credit/cs-training.csv")
test_pd = pd.read_csv("./data/give_credit/cs-test.csv")

# 是否显示图片
show_plt = False

# 替换变量名为中文
ch_name_dict = {
    'Unnamed: 0':'id',
    'SeriousDlqin2yrs':'好坏客户',
    'RevolvingUtilizationOfUnsecuredLines':'可用额度比值', 
    'age':'年龄',
    'NumberOfTime30-59DaysPastDueNotWorse':'逾期30-59天笔数',
    'DebtRatio':'负债率',
    'MonthlyIncome':'月收入',
    'NumberOfOpenCreditLinesAndLoans':'信贷数量',
    'NumberOfTimes90DaysLate':'逾期90天笔数',
    'NumberRealEstateLoansOrLines':'固定资产贷款量',
    'NumberOfTime60-89DaysPastDueNotWorse':'逾期60-89天笔数',
    'NumberOfDependents':'家属数量'
}

# 替换表头为中文    
def replace_header():
    global train_pd, test_pd
    train_pd.rename(columns=ch_name_dict, inplace=True)
    train_pd = train_pd.drop(['id'], axis=1)
    test_pd.rename(columns=ch_name_dict, inplace=True)
    test_pd = test_pd.drop(['id'], axis=1)


# 数据集总览
def data_overview():
    global train_pd, test_pd
    print('trainging data with new header, shape:', train_pd.shape)
    train_pd.info()
    print(train_pd.describe())
    # 保存到 csv 中
    train_pd.describe().to_csv('./data/give_credit/cs-training-desc.csv')
    print('-'*20)
    print('testing data with new header, shape:', test_pd.shape)
    test_pd.info()
    print(test_pd.describe())
    test_pd.describe().to_csv('./data/give_credit/cs-test-desc.csv')


# 查看各个变量的缺失比例
def missing_analysis():
    values = list(ch_name_dict.values())
    for key in values[1:]:
        ratio = train_pd[key].isnull().sum() / train_pd.shape[0]
        if ratio == 0.0:
            print(f'【{key}】字段无缺失')
        else:
            print(f'【{key}】缺失比: {ratio:.2%}')

# 处理缺失值，支持不同的模式
def missing_process(mode=0):
    global train_pd
    if mode == 0:
        print('月收入补均值，家属数量直接删除缺失样本')
        key = '月收入'
        # 填补均值
        train_pd[key] = train_pd[key].fillna(train_pd[key].mean())
        # 删除缺失值
        train_pd = train_pd.dropna()
    elif mode == 1:
        from sklearn.ensemble import RandomForestRegressor
        print('月收入采用随机森林法补值，家属数量直接删除缺失样本')
        # 把已有的数值型特征都列出来，其中因为要给月收入插值，所以放到第一个
        # 11 是家属数量，有缺失值，丢掉
        # 0 是 index，丢掉
        # 6 对应的是月收入
        process_df = train_pd.ix[:, [5, 1, 2, 3, 4, 6, 7, 8, 9]]
        # 分成已知月收入的部分和未知月收入的部分，并进行预测
        key = '月收入'
        known = process_df[process_df[key].notnull()].as_matrix()
        unknown = process_df[process_df[key].isnull()].as_matrix()
        # X 为特征，y 为 label
        X = known[:, 1:]
        y = known[:, 0]
        rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
        rfr.fit(X, y)
        # 用模型预测未知值
        preds = rfr.predict(unknown[:, 1:]).round(0)
        train_pd.loc[train_pd[key].isnull(), key] = preds
        # 删除缺失值
        train_pd = train_pd.dropna()

    print('缺失值处理完后的数据集')
    train_pd.info()

# 处理异常值
def abnormal_process(mode=0):
    global train_pd
    if mode == 0:
        # mode 0: 通过 四分位数来过滤异常值
        values = list(ch_name_dict.values())
        skip_set = set(['好坏客户'])
        k = 1.5
        for key in values[1:]:
            if key in skip_set:
                print(f'特征【{key}】在忽略列表中，不处理')
                continue
            print(f'开始过滤特征【{key}】中的异常值')
            q1 = train_pd[key].quantile(0.25)
            q3 = train_pd[key].quantile(0.75)
            iqr = q3 - q1
            low = q1 - k * iqr
            high = q3 + k + iqr
            print(f'q1:{q1},q3:{q3},iqr:{iqr},low:{low},high:{high}')
            train_pd = train_pd[(train_pd[key] >= low) & (train_pd[key] <= high)] 
    elif mode == 1:
        # mode 1: 人工观察箱线图和基于常识进行过滤
        train_pd = train_pd[train_pd['年龄'] > 0]  
        train_pd = train_pd[train_pd['逾期30-59天笔数'] < 90]    

    print('再次去掉重复值')
    train_pd = train_pd.drop_duplicates()
    print('过滤异常值后的数据集')
    train_pd.info()
    train_pd.describe()

# 进行探索性分析，在 mac 上会有问题，看看 seaborn 怎么样
def eda_process():
    global train_pd
    print('各个变量的直方图')
    train_pd.hist(figsize=(20, 15))
    if show_plt:
        plt.show()

    print('分析变量间关系')
    corr = train_pd.corr() # 计算相关性系数
    ylabel = list(corr.index)
    xlabel = ylabel
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    import seaborn
    seaborn.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size':9, 'weight': 'bold', 'color': 'blue'})
    ax1.set_xticklabels(xlabel, rotation=1, fontsize=8)
    ax1.set_yticklabels(ylabel, rotation=0, fontsize=10)
    if show_plt:
        plt.show()


# 具体的分箱操作，采用最优分段
import scipy.stats
def binning_op(y, x, n=12):
    # x 是要分箱的变量， y 是 label，n 是分组个数（不断试验）
    r = 0 # 相关系数
    bad = y.sum()
    good = y.count() - bad
    show = False
    d2 = None
    while np.abs(r) < 1: # 相关系数为 1 的时候停止
        # qcut 根据值的频率选择分箱的均匀间隔
        d1 = pd.DataFrame({"x": x, "y": y, "bucket": pd.qcut(x, n)})
        if show:
            print('d1 head ----------')
            print(d1.head(10))
        # 按照 bucket 字段来分组，并且会直接作为索引    
        d2 = d1.groupby('bucket', as_index=True)
        # 计算 spearmanr 等级相关系数和 p 值    
        r, p = scipy.stats.spearmanr(d2.mean().x, d2.mean().y)
        n = n -1
        show = False
    
    d3 = pd.DataFrame(d2.x.min(), columns=['min'])
    d3['min'] = d2.min().x # 分组最小值
    d3['max'] = d2.max().x # 分组最大值
    d3['sum'] = d2.sum().y # 对应分组的坏客户数量
    d3['total'] = d2.count().y # 对应分组的总客户数量
    d3['rate'] = d2.mean().y # 对应分组坏客户比例
    # 这里的计算和博客中有一点差别，两种都可以，这种相对来说更方便
    d3['woe'] = np.log((d3['rate']/(1-d3['rate'])) / (bad/good) ) # 计算 WOE
    d3['iv'] = (d3['sum']/bad - (d3['total'] - d3['sum'])/good) * d3['woe']

    d4 = (d3.sort_values(by='min')).reset_index(drop=True) # 把 min 重新变成数据列
    print('-'*20)
    print(d4)

    cut = []
    cut.append(float('-inf'))
    for i in range(1, n+1):
        qua = x.quantile(i/(n+1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))

    return d4, cut, woe

# 针对不能采用最优分箱的变量，用等频分箱
def cut_op(y, x, cut):
    bad = y.sum()
    good = y.count() - bad

    d1 = pd.DataFrame({"x": x, "y": y, "bucket": pd.cut(x, cut)})
    d2 = d1.groupby('bucket', as_index=True) # 分组聚合
    d3 = pd.DataFrame(d2.x.min(), columns=['min'])
    
    d3['min'] = d2.min().x # 分组最小值
    d3['max'] = d2.max().x # 分组最大值
    d3['sum'] = d2.sum().y # 对应分组的坏客户数量
    d3['total'] = d2.count().y # 对应分组的总客户数量
    d3['rate'] = d2.mean().y # 对应分组坏客户比例
    # 这里的计算和博客中有一点差别，两种都可以，这种相对来说更方便
    d3['woe'] = np.log((d3['rate']/(1-d3['rate'])) / (bad/good) ) # 计算 WOE
    d3['iv'] = (d3['sum']/bad - (d3['total'] - d3['sum'])/good) * d3['woe']

    d4 = (d3.sort_values(by='min')).reset_index(drop=True) # 把 min 重新变成数据列
    print('-'*20)
    print(d4)
    woe = list(d4['woe'].round(3))
    return d4, woe

# 计算各个分箱的 WOE 和 IV
def get_woe_iv():
    values = list(ch_name_dict.values())
    key_y = values[1]
    optimal_list = [2, 3, 5, 6] # 最优分箱
    cut_index = [4, 7, 8, 9, 10, 11] # 自定义分箱
    x_list = [None] * len(values)
    cut_list = [None] * len(values)
    woe_list = [None] * len(values)

    for idx in optimal_list:
        print(f'为特征【{values[idx]}】进行最优分箱')
        x_list[idx], cut_list[idx], woe_list[idx] = binning_op(train_pd[key_y], train_pd[values[idx]])
    
    ninf = float('-inf') #负无穷大
    pinf = float('inf') #正无穷大
    cut_list[4] = [ninf, 0, 1, 3, 5, pinf]
    cut_list[7] = [ninf, 1, 2, 3, 5, pinf]
    cut_list[8] = [ninf, 0, 1, 3, 5, pinf]
    cut_list[9] = [ninf, 0, 1, 2, 3, pinf]
    cut_list[10] = [ninf, 0, 1, 3, pinf]
    cut_list[11] = [ninf, 0, 1, 2, 3, 5, pinf]

    for idx in cut_index:
        print(f'为特征【{values[idx]}】进行等频分箱')
        x_list[idx], woe_list[idx] = cut_op(train_pd[key_y], train_pd[values[idx]], cut_list[idx])

    print('分析 WOE 值')
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for idx in range(2, len(values)):
        x = int((idx - 2) / 4)
        y = (idx - 2) % 4
        # print(x, y)
        x_list[idx].woe.plot(ax=axes[x, y], title=values[idx])
    if show_plt:
        plt.show()

    print('分析 IV 值')
    iv_list = [item.iv.sum() for item in x_list[2:]]
    item_dict = dict(zip(values[2:], iv_list))
    iv_pd = pd.DataFrame(item_dict, index=[0])
    iv_plot = iv_pd.plot.bar(figsize=(15, 10))
    iv_plot.set_title('特征变量 IV 值')
    if show_plt:
        plt.show()

    print('选出 IV 值小于 0.1 的变量')
    for key, value in item_dict.items():
        if value < 0.1:
            print(f'需要过滤特征【{key}】，IV 值为 {value}')

    return cut_list, woe_list

# 将某一列的原始变量替换成 woe 权重
def replace_woe(series, cut, woe):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(woe[m])
        i += 1
    return list

# 将变量替换成 woe
def transfer_woe(cut_list, woe_list):
    print('将原始值替换成 woe 值')
    global train_pd, test_pd
    values = list(ch_name_dict.values())
    for idx in range(2, len(values)):
        v = values[idx]
        print(f'正在处理特征【{v}】', len(train_pd[v].values))
        train_pd[v] = replace_woe(train_pd[v].values, cut_list[idx], woe_list[idx])

    train_pd.info()
    train_pd.describe()
    
    train_pd.to_csv('./data/give_credit/cs-training-woe.csv', index=False)

# 构建模型
def build_model(data_path):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from sklearn.linear_model import LogisticRegression
    print('开始构建模型，数据源', data_path)
    data = pd.read_csv(data_path)
    Y = data['好坏客户']
    X = data.drop(['好坏客户', '负债率', '月收入', '信贷数量', '固定资产贷款量', '家属数量'], axis=1)
    # 切分训练集和验证集
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=233)
    print('Train X Count: ', len(train_x), ' Shape: ', train_x.shape)
    print('Test X Count: ', len(test_x), ' Shape: ', test_x.shape)

    clf = LogisticRegression()
    clf.fit(train_x, train_y)
    score_proba = clf.predict_proba(test_x)
    y_predproba = score_proba[:,1]
    # 取出系数
    coe = clf.coef_[0]
    c = clf.intercept_

    print('模型结果')
    print('特征系数', coe)
    print('截距', c)

    print('测试集验证')
    #X2 = sm.add_constant(test_x)
    results = clf.predict_proba(test_x)
    y_predproba = results[:,1]
    fpr, tpr, threshold = roc_curve(test_y, y_predproba)
    rocauc = auc(fpr, tpr)
    # 计算 KS
    ks = 0.0
    for i in range(len(tpr)):
        gap = tpr[i] - fpr[i]
        if gap > ks:
            ks = gap
    print('Test AUC', rocauc)
    print('Test KS', ks * 100)
    plt.plot(fpr, tpr, 'b', label=f'AUC = {rocauc:.2f}') # 生成 ROC 曲线
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR 真正率')
    plt.xlabel('FPR 假正率')
    if show_plt:
        plt.show()
    return clf, coe, c

# 计算各个变量的评分卡
def get_score(coe, woe, B):
    score = []
    #print('coe', coe)
    #print('woe', woe)
    for w in woe:
        a = round(B * coe * w, 0)
        score.append(a)
    return score

# 生成评分卡
def generate_card(model, coe, c, woe_list, cut_list):
    print('生成评分卡，基础分为 600，PDO 为 20，违约与正常的比例是 1/60')
    B = 20 / np.log(2)
    A = 600 + B*np.log(1/60)
    print('A', A)
    print('B', B)
    base_score = A - B * c
    print('基础分值', base_score)

    values = list(ch_name_dict.values())
    score_list = [None] * len(values)
    # 这里要索引加一位
    variable_list = [2, 3, 4, 8, 10]
    c_i = 0
    for idx in variable_list:
        score = get_score(coe[c_i], woe_list[idx], B)
        print(f'特征【{values[idx]}】评分卡')
        cuts = cut_list[idx]
        for i in range(len(cuts)-1):
            print(f'({cuts[i]},{cuts[i+1]}] {score[i]}')
        score_list[idx] = score
        c_i = c_i + 1

    # 返回评分卡
    return score_list, base_score[0]

def compute_score(series, cut, score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list

# 给用户进行评分
def rank_user(base_score, cut_list, score_list):
    global test_pd
    values = list(ch_name_dict.values())
    # 这里要索引加一位
    variable_list = [2, 3, 4, 8, 10]
    export_list = ['基础分值', '最终评分']
    test_pd['基础分值'] = pd.Series(np.zeros(len(test_pd)) + base_score)
    test_pd['最终评分'] = test_pd['基础分值']
    for idx in variable_list:
        print(f'正在为特征【{values[idx]}】评分')
        export_list.append(f'{values[idx]}_评分')
        test_pd[f'{values[idx]}_评分'] = pd.Series(compute_score(test_pd[values[idx]], cut_list[idx], score_list[idx]))
        test_pd['最终评分'] += test_pd[f'{values[idx]}_评分']
    print('正在输出最终评分结果')
    test_pd[export_list].to_csv('./data/give_credit/cs-testing-score.csv', index=False)

if __name__ == "__main__":
    etl_switch = True
    woe_switch = True
    model_switch = True
    pred_switch = True

    if etl_switch:
        replace_header()
        data_overview()
        missing_analysis()

    woe_list, cut_list = [], []
    if woe_switch:
        mode = 1
        missing_process(mode=mode)
        abnormal_process(mode=mode)
        eda_process()
        cut_list, woe_list = get_woe_iv()
        transfer_woe(cut_list, woe_list)
    
    score_list = []
    base_score = 0
    if model_switch:
        data_path = './data/give_credit/cs-training-woe.csv'
        model, coe, c = build_model(data_path)
        score_list, base_score = generate_card(model, coe, c, woe_list, cut_list)

    if pred_switch:
        rank_user(base_score, cut_list, score_list)



    

    