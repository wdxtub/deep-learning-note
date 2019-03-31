import tensorflow as tf
import numpy as np
import threading
import time


def MyLoop(coord, worker_id):
    # 判断当前线程是否要停止
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stoping from id: %d\n" % worker_id)
            # 通知其他线程停止
            coord.request_stop()
        else:
            print("Working on id: %d\n" % worker_id)
        time.sleep(1)


# coord 协同多个线程
coord = tf.train.Coordinator()
# 创建 5 个线程
threads = [ threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5) ]

for t in threads:
    t.start()
# 等待所有线程退出
coord.join(threads)
