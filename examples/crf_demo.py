# -*- encoding: utf-8 -*-
'''
@Email :  wushaojun0107@gmail.com
@Time  :  02-05 2021
'''

# 用MEMM CRF 做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
import os
import sys
sys.path[0] = os.getcwd()
from bert4one.backend import K
from bert4one.build import build_bert_model
from bert4one.tokenizer import Tokenizer
from bert4one.optimizers import Adam
from bert4one.layers import ConditionalRandomField, MaximumEntropyMarkovModel
from bert4one.snippets import NerDataGenerator, NerEvaluator, NamedEntityRecognizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from examples.config import Config



config = Config()

maxlen = 256
epochs = 10
batch_size = 32
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
memm_lr_multiplier = 1000  # 必要时扩大MEMM层的学习率


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, this_flag = c.split(' ')
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


# 标注数据
train_data = load_data(config.china_daily_train)[:100]
valid_data = load_data(config.china_daily_dev)[:10]
test_data = load_data(config.china_daily_test)[:10]

# 建立分词器
tokenizer = Tokenizer(config.vocab_path, do_lower_case=True)

# 类别映射
classes = set(['PER', 'LOC', 'ORG'])
id2class = dict(enumerate(classes))
class2id = {j: i for i, j in id2class.items()}
num_labels = len(classes) * 2 + 1


model = build_bert_model(
    config.config_path,
    config.checkpoint_path,
    model=config.model[config.model_index]
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers-1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)

CRF = MaximumEntropyMarkovModel(lr_multiplier=memm_lr_multiplier)
result = CRF(output, mask=None)

model = Model(model.input, result)
model.summary()

model.compile(loss=CRF.sparse_loss,
              optimizer=Adam(learing_rate),
              metrics=[CRF.sparse_accuracy])


NER = NamedEntityRecognizer(
    id2class, trans=K.eval(CRF.trans), starts=[0], ends=[0])
evaluator = NerEvaluator(valid_data, CRF, NER,
                         tokenizer, config.save_model_path)
train_generator = NerDataGenerator(
    tokenizer, maxlen, class2id, train_data, batch_size)

model.fit(train_generator.generate(),
          steps_per_epoch=len(train_generator),
          epochs=epochs,
          callbacks=[evaluator])
