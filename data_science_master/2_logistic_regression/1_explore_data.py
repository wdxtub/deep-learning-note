# -*- coding: UTF-8 -*-
import pandas as pd

data = pd.read_csv('data/adult.data')
cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'label']
data = data[cols]
print '读取数据'
print data.head(8)
print '数据转换'
data['label_code'] = pd.Categorical(data.label).codes
print data[['label', 'label_code']].head(8)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
data[['age', 'hours_per_week', 'education_num', 'label_code']].hist()

plt.show(block=False)

print '数据基本统计信息'
print data.describe()

print '基本统计信息'
cross1 = pd.crosstab(pd.qcut(data['education_num'], [0, .25, .5, .75, 1]), data['label'])
print cross1

from statsmodels.graphics.mosaicplot import mosaic
mosaic(cross1.stack())

cross2 = pd.crosstab(pd.cut(data['hours_per_week'], 5), data['label'])
# 交叉报表归一化，利于分析统计
cross2_norm = cross2.div(cross2.sum(1).astype(float), axis=0)
cross2_norm.plot(kind='bar')
plt.show(block=False)