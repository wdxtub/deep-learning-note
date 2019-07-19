# 创建 Thread 的实例，传给它一个函数

import threading
from time import sleep, ctime

loops = [4, 2]


def loop(nloop, nsec):
    print('Start LOOP', nloop, 'at:', ctime())
    sleep(nsec)
    print('LOOP', nloop, 'DONE at:', ctime())


def main():
    print('程序开始，当前时间', ctime())
    threads = []
    nloops = range(len(loops))

    for i in nloops:
        t = threading.Thread(target=loop, args=(i, loops[i]))
        threads.append(t)

    for i in nloops:
        threads[i].start()

    for i in nloops:
        threads[i].join() # 等待进程完成

    print('程序结束，当前时间', ctime())


if __name__ == '__main__':
    main()
