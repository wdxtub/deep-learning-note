import torch
from torch import nn, optim
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('载入数据')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = utils.load_data_jay_lyrics()

print('初始化模型参数')
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

print('直接使用 nn.LSTM')
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = utils.RNNModel(lstm_layer, vocab_size).to(device)

print('训练并创作歌词')
num_epochs, num_steps, batch_size, lr, clipping_theta = 300, 35, 32, 1e-2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['可爱女人', '龙卷风']
utils.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                    corpus_indices, idx_to_char, char_to_idx,
                                    num_epochs, num_steps, lr, clipping_theta,
                                    batch_size, pred_period, pred_len, prefixes)

'''
训练并创作歌词
epoch 40, perplexity 1.020403, time 2.11 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱
 - 龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受
epoch 80, perplexity 1.015161, time 2.00 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱
 - 龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受
epoch 120, perplexity 1.020574, time 2.01 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱
 - 龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受
epoch 160, perplexity 1.016605, time 2.03 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱
 - 龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受
epoch 200, perplexity 1.015813, time 2.02 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱
 - 龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受
epoch 240, perplexity 1.025380, time 2.08 sec
 - 可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受
epoch 280, perplexity 1.010178, time 1.98 sec
 - 可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受
'''