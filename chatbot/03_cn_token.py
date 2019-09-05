import pynlpir

s = '今天天气真不错，但是晚上要开会。'

pynlpir.open()
print('----------------')
print('普通分词')
print('----------------')
segments = pynlpir.segment(s)
for segment in segments:
    print(segment[0], '\t', segment[1])

print('----------------')
print('提取关键词')
print('----------------')
key_words = pynlpir.get_key_words(s, weighted=True)
for key_word in key_words:
    print(key_word[0], '\t', key_word[1])

print('----------------')
print('中文分析')
print('----------------')
segments = pynlpir.segment(s, pos_names='all', pos_english=False)
for segment in segments:
    print(segment[0], '\t', segment[1])
pynlpir.close()
