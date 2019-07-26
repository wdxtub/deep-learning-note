# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何使用statsmodels搭建线性回归模型
"""

import os, sys
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import pandas as pd


def modelSummary(re):
    """
    分析线性回归模型的统计性质
    """
    # 整体统计分析结果
    print(re.summary())
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    # 用f test检测x对应的系数a是否显著
    print("检验假设x的系数等于0：")
    print(re.f_test("x=0"))
    # 用f test检测常量b是否显著
    print("检测假设const的系数等于0：")
    print(re.f_test("const=0"))
    # 用f test检测a=1, b=0同时成立的显著性
    print("检测假设x的系数等于1和const的系数等于0同时成立：")
    print(re.f_test(["x=1", "const=0"]))


def trainModel(X, Y):
    """
    训练模型
    """
    model = sm.OLS(Y, X) # 创建一个线性模型
    re = model.fit()
    return re

def linearModel(data):
    """
    线性回归统计性质分析步骤展示

    参数
    ----
    data : DataFrame，建模数据
    """
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    # 加入常量变量
    X = sm.add_constant(data[features])
    # 构建模型
    re = trainModel(X, Y)
    # 分析模型效果
    modelSummary(re)
    # const并不显著，去掉这个常量变量
    resNew = trainModel(data[features], Y)
    # 输出新模型的分析结果
    print(resNew.summary())

def readData(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    return data

if __name__ == "__main__":
    data = readData("data/simple_example.csv")
    linearModel(data)