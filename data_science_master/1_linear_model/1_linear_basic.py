# -*- coding: UTF-8 -*-
"""
此脚本用于展示使用sklearn搭建线性回归模型
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn import linear_model

def evaluateModel(model, testData, features, labels):
    """
    计算线性模型的均方差和决定系数

    参数
    ----
    model : LinearRegression, 训练完成的线性模型

    testData : DataFrame，测试数据

    features : list[str]，特征名列表

    labels : list[str]，标签名列表

    返回
    ----
    error : np.float64，均方差

    score : np.float64，决定系数
    """
    # 均方差(The mean squared error)，均方差越小越好
    error = np.mean(
        (model.predict(testData[features]) - testData[labels]) ** 2)
    # 决定系数(Coefficient of determination)，决定系数越接近1越好
    score = model.score(testData[features], testData[labels])
    return error, score

def trainModel(trainData, features, labels):
    """
    利用训练数据，估计模型参数

    参数
    ----
    trainData : DataFrame，训练数据集，包含特征和标签

    features : 特征名列表

    labels : 标签名列表

    返回
    ----
    model : LinearRegression, 训练好的线性模型
    """
    # 创建一个线性回归模型
    model = linear_model.LinearRegression()
    # 训练模型，估计模型参数
    model.fit(trainData[features], trainData[labels])
    return model

def linearModel(data):
    features = ["x"]
    labels = ["y"]
    
    train_data = data[:15]
    test_data = data[15:]

    model = trainModel(train_data, features, labels)
    error, score = evaluateModel(model, test_data, features, labels)
    print "error", error
    print "score", score


def readData(path):
    """
    使用pandas读取数据
    csv file
    x, y
    10, 7
    11, 8
    ...
    """
    data = pd.read_csv(path)
    return data

if __name__ == "__main__":
    data = readData("data/simple_example.csv")
    linearModel(data)