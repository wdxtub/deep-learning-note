from atexit import register
from random import randrange
from threading import currentThread, Lock, Thread
from time import ctime, sleep

# 使用锁的一个例子

class CleanOutputSet(set):
    def __str__(self):
        return ",".join(x for x in self)

lock = Lock()
loops = (randrange(2, 5) for x in range(randrange(3, 7)))
remaining = CleanOutputSet()


def loop(nsec):
    myname = currentThread().name
    with lock:
        remaining.add(myname)
        print("[{}] Started {}".format(ctime(), myname))
    sleep(nsec)
    with lock:
        remaining.remove(myname)
        print("[{}] Completed {} ({} secs)".format(ctime(), myname, nsec))
        print("    (remaining: {})".format(remaining or "None"))


def _main():
    for pause in loops:
        Thread(target=loop, args=(pause,)).start()


@register
def _atexit():
    print("all Done at", ctime())


if __name__ == '__main__':
    _main()
