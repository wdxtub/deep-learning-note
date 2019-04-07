import codecs
import collections
from operator import itemgetter

# 将文本转化为模型可以读入的单词序列，并保存到独立的 vocab 文件中
RAW_DATA = "../data/ptb/ptb.train.txt"
VOCAB_OUTPUT = "data/ptb.vocab"

counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

# 按照词频排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

# 将句子结束符 <eos> 放入词表
sorted_words = ["<eos>"] + sorted_words
# 注意，如果有需要，还要将 <unk> 和 <sos> 句子起始符加入词表，但 PTB 数据中已经把低频替换成 <unk>
# 所以不用处理

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")
            