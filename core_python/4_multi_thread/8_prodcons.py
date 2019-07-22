# 用 Queue 来实现生产者-消费者问题。生产者和消费者独立且并发执行线程

from myThread import MyThread
from queue import Queue
from random import randint
from time import sleep, ctime
from atexit import register


def writeQ(queue):
    print("producing object for Q...", end=" ")
    queue.put("xxx", 1)
    print("size now:", queue.qsize())


def readQ(queue):
    val = queue.get(1)
    print("consumed object from Q... size now:", queue.qsize())


def writer(queue, loops):
    for i in range(loops):
        writeQ(queue)
        sleep(randint(1, 3))


def reader(queue, loops):
    for i in range(loops):
        readQ(queue)
        sleep(randint(2, 5))


funcs = [writer, reader]
nfuncs = range(len(funcs))


def main():
    nloops = randint(2, 20)
    q = Queue(32)

    threads = []
    for i in nfuncs:
        t = MyThread(funcs[i], (q, nloops), funcs[i].__name__)
        threads.append(t)

    for i in nfuncs:
        threads[i].start()

    for i in nfuncs:
        threads[i].join()

    print("all DONE")

@register
def _atexit():
    print("all done at:", ctime())


if __name__ == "__main__":
    main()
