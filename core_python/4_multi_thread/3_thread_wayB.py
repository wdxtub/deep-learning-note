# 派生 Thread 的子类，并创建子类的实例，定制线程对象拥有更多灵活性，同时简化线程创建调用过程

import threading
from time import sleep, ctime

loops = (4, 2)

class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


def loop(nloop, nsec):
    print('Start LOOP', nloop, 'at:', ctime())
    sleep(nsec)
    print('LOOP', nloop, 'DONE at:', ctime())


def main():
    print('程序开始，当前时间', ctime())
    threads = []
    nloops = range(len(loops))

    for i in nloops:
        t = MyThread(loop, (i, loops[i]))
        threads.append(t)

    for i in nloops:
        threads[i].start()

    for i in nloops:
        threads[i].join()  # 等待进程完成

    print('程序结束，当前时间', ctime())


if __name__ == '__main__':
    main()
