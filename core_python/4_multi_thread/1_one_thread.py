# 在一个线程里执行两个循环，只能顺序执行

from time import sleep, ctime


def loop0():
    print('开始 Loop 0，当前时间', ctime())
    sleep(4)
    print('Loop 0 结束，当前时间', ctime())


def loop1():
    print('开始 Loop 1，当前时间', ctime())
    sleep(2)
    print('Loop 1 结束，当前时间', ctime())


def main():
    print('程序开始，当前时间', ctime())
    loop0()
    loop1()
    print('程序结束，当前时间', ctime())


if __name__ == '__main__':
    main()

