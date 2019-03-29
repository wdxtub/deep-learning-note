import tensorflow as tf
from tensorflow.python.client import timeline

a = tf.random_normal([2000, 5000])
b = tf.random_normal([5000, 1000])
res = tf.matmul(a, b)

with tf.Session() as sess:
    # 添加记录 session 执行的选项
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(res, options=options, run_metadata=run_metadata)

    # 创建 Timeline 对象，并写入到 json 文件中
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('trace/timeline_1_simple.json', 'w') as f:
        f.write(chrome_trace)

