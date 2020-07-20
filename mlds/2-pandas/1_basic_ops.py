import pandas as pd
import numpy as np

def split_line():
    print('--------------------')

print('创建 Series')
s = pd.Series([1, 2, 3, 4, np.nan, 6, 7])
print(s)
split_line()

print('创建日期 DataFrame')
dates = pd.date_range('20200314', periods=6)
print(dates)
split_line()

print('通过 numpy 创建 DataFrame')
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)
split_line()

print('通过 dict 创建 DataFrame')
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3]*4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })
print(df2)
print(df2.dtypes)
split_line()

print('查看头部几行')
print(df.head())
print('查看尾部几行')
print(df.tail())
split_line()

print('显示索引、列名及底层 numpy 数据')
print(df.index)
print(df.columns)
print(df.values)
split_line()

print('对数据进行快读统计')
print(df.describe())
split_line()

print('对数据进行转置')
print(df.T)
split_line()

print('按照列名排序')
print(df.sort_index(axis=1, ascending=False))
split_line()

print('按照某一列的值进行排序')
print(df.sort_values(by='B'))
split_line()

print('虽然标准的Python/Numpy的表达式能完成选择与赋值等功能，但我们仍推荐使用优化过的pandas数据访问方法：.at，.iat，.loc，.iloc和.ix')
print('')

print('选择某一列数据，返回 Series')
print(df['A'])
print('使用 [] 切片')
print(df[0:3])
split_line()

print('通过标签选取')
print(df.loc[dates[0]])
print('选取多列')
print(df.loc[:, ['A', 'C']])
print('行列同时选择')
print(df.loc['2020-03-14': '2020-03-16', ['A', 'C']])
print('快速获取某个值')
print(df.at[dates[0], 'D'])
split_line()

print('通过位置选取，直接传递整型')
print(df.iloc[3])
print('行列同时选择')
print(df.iloc[3:5, 0:2])
print('只选取行')
print(df.iloc[1:3, :])
print('只选取列')
print(df.iloc[:, 1:3])
print('取具体的值（两种方法）')
print(df.iloc[2,1], df.iat[2, 1])
split_line()

print('通过布尔索引取值，即通过判断过滤')
print('选取 A 列 >0')
print(df[df.A > 0])
print('选取 >0，小于 0 的会变成 NaN')
print(df[df > 0])
print('通过 isin() 过滤数据，主要针对字符串')
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print('before', df2)
print('after', df2[df2['E'].isin(['one', 'two'])])
split_line()

print('赋值一个新的列，通过索引来自动对齐数据')
s1 = pd.Series([1,2,3,4,5], index=pd.date_range('20200314',periods=5))
print(s1)
df['F'] = s1
print(df)
print('通过标签赋值')
df.at[dates[0], 'A'] = 0
print(df)
print('通过位置赋值')
df.iat[0, 1] = 0
print('通过 numpy 赋值')
df.loc[:, 'D'] = np.array([5]*len(df))
print(df)
print('通过 where 赋值')
df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)
split_line()

print('缺失值处理，在pandas中，用np.nan来代表缺失值，这些值默认不会参与运算')
df1 = df.reindex(index=dates[0:4], columns=list(df.columns)+['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
print(df1)
print('删除所有包含缺失值的行数据')
print(df1.dropna(how='any'))
print('填充缺失值')
print(df1.fillna(value=5))
print('获取值是否为nan的布尔标记')
print(pd.isnull(df1))
split_line()

print('按列求平均')
print(df.mean())
print('按行求平均')
print(df.mean(1))
split_line()

print('apply 函数默认会按列进行运算')
print('apply 按列累加')
print('before', df)
print('after', df.apply(np.cumsum))
print('apply 找到每列的差值')
print(df.apply(lambda x:x.max() - x.min()))
split_line()

print('频数统计')
s = pd.Series(np.random.randint(0, 7, size=10))
print('origin data', s)
print(s.value_counts())
split_line()

print('处理字符串')
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print('before', s)
print('after', s.str.lower())
split_line()

print('连接 Series，DataFrame 和 Panel 对象')
df = pd.DataFrame(np.random.randn(10,4))
print(df)
print('拆分成不同元素')
pieces = [df[:3], df[3:7], df[7:]]
print(pieces[0])
print('再合并起来')
print(pd.concat(pieces))
split_line()

print('Join 操作')
left = pd.DataFrame({'key':['foo', 'foo'], 'lval':[1,2]})
right = pd.DataFrame({'key':['bar', 'foo'], 'lval':[4,5]})
print('left', left)
print('right', right)
print('merge', pd.merge(left, right, on='key'))
split_line()

print('添加行到 DataFrame 后面')
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
print('before', df)
s = df.iloc[3]
print('after', df.append(s, ignore_index=True))
split_line()

print('分组操作，针对每组进行不同的计算，最后合并到某一个数据结构')
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar', 
                          'foo', 'bar', 'foo', 'bar'],
                   'B' : ['one', 'one', 'two', 'three', 
                          'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
print(df)
print('对 A 列进行 group by')
print(df.groupby('A').sum())
print('对 A 和 B 列进行 group by')
print(df.groupby(['A', 'B']).sum())
split_line()

print('数据透视表')
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})
print(df)
print('生成透视表')
print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))
split_line()

print('处理时间序列数据')
print('生成时间序列')
rng = pd.date_range('2020/03/14', periods=100, freq='S')
print(rng)
print('给每个时间点赋值')
ts = pd.Series(np.random.randint(0,500,len(rng)), index=rng)
print(ts)
print('重新取样')
print(ts.resample('1Min', how='sum'))
print('时区表示')
rng = pd.date_range('3/6/2012', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)
print('UTC 时间')
ts_utc = ts.tz_localize('UTC')
print(ts_utc)
print('时区转化')
print(ts_utc.tz_convert('US/Eastern'))
print('时间跨度转化')
rng = pd.date_range('1/1/2012', periods=5, freq='M')
print(rng)
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)
print(ts.to_period())
print(ts.to_period().to_timestamp())
split_line()

print('类别数据')
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'e', 'e']})
print(df)
print('将 raw_grade 转换成分类类型')
df["grade"] = df["raw_grade"].astype("category")
print(df["grade"])
print('重命名类别为更有意义的内容')
df["grade"].cat.categories = ["very good", "good", "very bad"]
print('对分类重新排序，并添加缺失的分类')
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
print(df["grade"])
print('排序是按照分类的顺序进行的，而不是字典序')
print(df.sort_values(by="grade"))
print('按分类分组时，也会显示空的分类')
print(df.groupby("grade").size())