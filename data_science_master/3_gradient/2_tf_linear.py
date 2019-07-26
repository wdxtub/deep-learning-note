# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import math


def generateLinearData(dimension, num):
    np.random.seed(1024)
    beta = np.array(range(dimension)) + 1
    x = np.random.random((num, dimension))
    epsilon = np.random.random((num, 1))
    # 将被预测值写成矩阵形式，会极大加快速度
    y = x.dot(beta).reshape((-1, 1)) + epsilon
    return x, y


def createLinearModel(dimension):
    np.random.seed(1024)
    # 定义 x 和 y
    x = tf.placeholder(tf.float64, shape=[None, dimension], name='x')
    # 写成矩阵形式会大大加快运算速度
    y = tf.placeholder(tf.float64, shape=[None, 1], name='y')
    # 定义参数估计值和预测值
    betaPred = tf.Variable(np.random.random([dimension, 1]))
    yPred = tf.matmul(x, betaPred, name='y_pred')
    # 定义损失函数
    loss = tf.reduce_mean(tf.square(yPred - y))
    model = {
        'loss_function': loss,
        'independent_variable': x,
        'dependent_variable': y,
        'prediction': yPred,
        'model_params': betaPred
    }
    return model


def stochasticGradientDescent(X, Y, model, learningRate=0.01,
                              miniBatchFraction=0.01, epoch=10000, tol=1.e-6):
    method = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    optimizer = method.minimize(model['loss_function'])

    # 增加日志
    tf.summary.scalar('loss_function1', model['loss_function'])
    tf.summary.histogram('params1', model['model_params'])
    tf.summary.scalar('first_param1', tf.reduce_mean(model['model_params'][0]))
    tf.summary.scalar('last_param1', tf.reduce_mean(model['model_params'][-1]))
    summary = tf.summary.merge_all()
    # 程序运行结束后执行 tensorboard --logdir logs/
    summaryWriter = createSummaryWriter('logs/sto_gradient_descent')


    # TF 开始运行
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # 迭代梯度下降法
    step = 0
    batchSize = int(X.shape[0] * miniBatchFraction)
    batchNum = int(math.ceil(1 / miniBatchFraction))
    prevLoss = np.inf
    diff = np.inf
    # 当损失函数的变动小于阈值或达到最大循环次数，则停止迭代
    while (step < epoch) & (diff > tol):
        for i in range(batchNum):
            # 选取小批次训练数据
            batchX = X[i * batchSize: (i+1) * batchSize]
            batchY = Y[i * batchSize: (i+1) * batchSize]
            # 迭代模型参数
            sess.run([optimizer],
                     feed_dict={
                         model['independent_variable']: batchX,
                         model['dependent_variable']: batchY
                     })
            # 计算损失函数

            _, summaryStr, loss = sess.run(
                [summary, model['loss_function']],
                feed_dict={
                    model['independent_variable']: X,
                    model['dependent_variable']: Y
                }
            )
            summaryWriter.add_summary(summaryStr, step)
            # 计算损失函数的变动
            diff = abs(prevLoss - loss)
            prevLoss = loss
            if diff <= tol:
                break
        step += 1



def gradientDescent(X, Y, model, learningRate=0.01, maxIter=10000, tol=1.e-6):
    # 确定最优算法
    method = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    optimizer = method.minimize(model['loss_function'])
    # 增加日志
    tf.summary.scalar('loss_function', model['loss_function'])
    tf.summary.histogram('params', model['model_params'])
    tf.summary.scalar('first_param', tf.reduce_mean(model['model_params'][0]))
    tf.summary.scalar('last_param', tf.reduce_mean(model['model_params'][-1]))
    summary = tf.summary.merge_all()
    # 程序运行结束后执行 tensorboard --logdir logs/
    summaryWriter = createSummaryWriter('logs/gradient_descent')

    # TF 开始运行
    sess = tf.Session()
    # 产生初始参数
    init = tf.global_variables_initializer()
    sess.run(init)

    # 迭代梯度下降
    step = 0
    prevLoss = np.inf
    diff = np.inf
    # 当损失函数的变动小于阈值或达到最大循环次数，则停止迭代
    while (step < maxIter) & (diff > tol):
        _, summaryStr, loss = sess.run(
            [optimizer, summary, model['loss_function']],
            feed_dict={
                model['independent_variable']: X,
                model['dependent_variable']: Y
            }
        )
        summaryWriter.add_summary(summaryStr, step)
        # 计算损失函数的变动
        diff = abs(prevLoss - loss)
        prevLoss = loss
        step += 1
    summaryWriter.close()


def createSummaryWriter(logPath):
    # 检查是否有老的文件，有的话清理
    if tf.gfile.Exists(logPath):
        tf.gfile.DeleteRecursively(logPath)
    summaryWriter = tf.summary.FileWriter(logPath, graph=tf.get_default_graph())
    return summaryWriter



def run():
    # 自变量个数
    dimension = 30
    num = 10000
    # 随机产生模型数据
    X, Y = generateLinearData(dimension, num)
    # 定义模型
    model = createLinearModel(dimension)
    # 使用梯度下降估计模型参数
    gradientDescent(X, Y, model)


if __name__ == '__main__':
    run()

