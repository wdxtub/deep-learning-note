import numpy as np
import scipy 

from scipy import constants as C

def split_line():
    print('--------------------')

print('scipy version', scipy.__version__)
print('常数和特殊函数')
print('真空中的光速', C.c)
print('普朗克常数', C.h)
print('1 英里等于多少米', C.mile)
print('1 英寸等于多少米', C.inch)
print('1 克等于多少千克', C.gram)
print('1 磅等于多少千克', C.pound)
split_line()

print('非线性方程组求解')
