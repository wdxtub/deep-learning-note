import tensorflow as tf

# 说明 softmax_cross_entropy_with_logtis 和 sparse_softmax_cross_entropy_with_logits 的用法区别

# 假设词汇表的大小为 3，语料包含两个单词 “2， 0”
word_labels = tf.constant([2, 0])

# 假设模型对两个单词进行预测时，产生的 logits 如下，注意，这里并不是概率
# 如果需要概率，则要调用 prob=tf.nn.softmax(logits)
predict_logits = tf.constant([2,0, -1.0, 3.0], [1.0, 0.0, -0.5])
# 使用 sparse_softmax_cross_entropy_with_logits 计算交叉熵
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels, logits=predict_logits)

# 运行并计算 loss，对应于这两个预测的 perplexity 损失
sess = tf.Session()
sess.run(loss)