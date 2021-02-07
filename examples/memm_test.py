# -*- encoding: utf-8 -*-
'''
@Email :  wushaojun0107@gmail.com
@Time  :  02-05 2021
'''

# 用MEMM做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
import os,sys
sys.path[0] = os.getcwd()
import numpy as np
import tensorflow as tf
print(tf.__version__)
from bert4one.backend import keras, K
print(keras.__version__)
from bert4one.build import build_bert_model
from bert4one.tokenizer import Tokenizer
from bert4one.optimizers import Adam
from bert4one.snippets import sequence_padding, DataGenerator
from bert4one.nlps.memm import MaximumEntropyMarkovModel
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tqdm import tqdm
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
train_data = load_data(config.china_daily_train)
valid_data = load_data(config.china_daily_dev)
test_data = load_data(config.china_daily_test)

# 建立分词器
tokenizer = Tokenizer(config.vocab_path, do_lower_case=True)

# 类别映射
classes = set(['PER', 'LOC', 'ORG'])
id2class = dict(enumerate(classes))
class2id = {j: i for i, j in id2class.items()}
num_labels = len(classes) * 2 + 1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in self.data[i]:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = class2id[l] * 2 + 1
                        I = class2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_bert_model(
    config.config_path,
    config.checkpoint_path,
    model=config.model[config.model_index]
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers-1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)

print(K.is_keras_tensor(output))
print(output)

MEMM = MaximumEntropyMarkovModel(lr_multiplier=memm_lr_multiplier)
result = MEMM(output, mask='SequenceMask')

model = Model(model.input, result)
model.summary()

model.compile(loss=MEMM.sparse_loss,
              optimizer=Adam(learing_rate),
              metrics=[MEMM.sparse_accuracy])


def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[0].argmax()]


def named_entity_recognize(text):
    """命名实体识别函数
    """
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 512:
        tokens.pop(-2)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(MEMM.trans)
    labels = viterbi_decode(nodes, trans)[1:-1]
    entities, starting = [], False
    for token, label in zip(tokens[1:-1], labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                entities.append([[token], id2class[(label - 1) // 2]])
            elif starting:
                entities[-1][0].append(token)
            else:
                starting = False
        else:
            starting = False
    return [(tokenizer.decode(w, w).replace(' ', ''), l) for w, l in entities]


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(named_entity_recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(MEMM.trans)
        print(trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(os.path.join(config.save_model_path,'best_model.weights'))
        print('valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
              (f1, precision, recall, self.best_val_f1))
        f1, precision, recall = evaluate(test_data)
        print('test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
              (f1, precision, recall))


if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(train_data, batch_size)

    model.fit(train_generator.generate(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])

else:

    model.load_weights('./best_model.weights')