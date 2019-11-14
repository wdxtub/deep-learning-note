
import torch
from torch import nn, optim
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('载入数据')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = utils.load_data_jay_lyrics()

print('初始化模型参数')
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

print('直接使用 nn.GRU')
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = utils.RNNModel(gru_layer, vocab_size).to(device)

print('训练并创作歌词')
num_epochs, num_steps, batch_size, lr, clipping_theta = 300, 35, 32, 1e-2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['可爱女人', '龙卷风']
utils.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                    corpus_indices, idx_to_char, char_to_idx,
                                    num_epochs, num_steps, lr, clipping_theta,
                                    batch_size, pred_period, pred_len, prefixes)


'''
训练并创作歌词
epoch 40, perplexity 1.037138, time 1.58 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱
 - 龙卷风 不能承受我已无处可躲 我不要再想 我不要再想 我不 我不 我不要再想你 爱情来的太快就像龙卷风 离
epoch 80, perplexity 1.805499, time 1.63 sec
 - 可爱女人睡着头发现 你说在墙上剥落的红色油漆 还没有多难熬 你说啊 你说啊 是因为喝醉 这样打球变得好笑不好
 - 龙卷风什么刀枪跟棍棒球 只对暴力一直伴奏 黑色帘幕被风吹动地方的星 一直走 一直一个人都遇见 连隔壁邻居都
epoch 120, perplexity 1.146178, time 1.81 sec
 - 可爱女人在着掉你的天 面放纵 那么凶 如果真的天 我要再想你这样打我妈的在这里是逃避 没人帮着你走才快乐是没
 - 龙卷风隐隐他的出土太多太多是几个人海一回忆 相思寄红豆无就带你和汉堡 过往的欢乐是否褪色想回到当年的躺在那
epoch 160, perplexity 1.066869, time 1.80 sec
 - 可爱女人有着放弃 你打再重 也会慢睡着 你在黑白的梦里 你身着的画面是你要的天 你你也接受他 不想太多难熬
 - 龙卷风暴力自由想你一步望著天 灵魂序曲完成 那大地心愿 木炭 你永远 都想这样好多 想给的你手 在隔壁看着
epoch 200, perplexity 1.022936, time 1.76 sec
 - 可爱女人有着放大 我怎么停留难承到 好难承受 荣耀的背后刻着一道孤独 仁慈的父我已坠入 就是那么简单你说啊
 - 龙卷风暴力自由想你只能看见你当年就飘开 整个人帮你擦眼泪 记得那梦的画面  这样的出生 在小村外的溪边河口
epoch 240, perplexity 1.015995, time 1.93 sec
 - 可爱女人睡着头发 在你身边 等到放晴的你有多难堪 我不懂 你怎么抄我球 你怎么打我手 你说啊 只想回到过去
 - 龙卷风暴力太多爱可不可不再多爱你在我连隔壁邻居都猜到哪里都是一点头看这书的你会听 一直放 我不我 的甜蜜
epoch 280, perplexity 1.014075, time 1.71 sec
 - 可爱女人睡着不屈 只是我怕眼泪撑不住你 不想是你开身为分手牵手你手牵手牵手都赢不了 牵着你的手过的沉会有原只
 - 龙卷风也不太太不能不去也布满弹孔的太多太多是你心中你只会值得去做家枪跟棍棒 我害怕你只会有泥鳅一只灰狼人称
'''