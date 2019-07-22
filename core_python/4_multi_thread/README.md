# 多线程

全局解释器锁 GIL 用来保证同时只有一个线程在运行

推荐使用 threading 模块而不是 thread，因为 thread 的同步原语较少（只有一个），并且无法控制何时退出线程

由于 Python 的 GIL 限制，多线程更适合于 I/O 密集型应用，而不是计算密集型。对于计算量大的任务来说，建议使用多进程，以便让 CPU 的其他内核来执行。

threading 模块的主要替代品：

1. subprocess 模块
2. multiprocessing 模块
3. concurrent.futures 模块