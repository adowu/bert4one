# -*- encoding: utf-8 -*-
'''
@Email :  wushaojun0107@gmail.com
@Time  :  02-07 2021
'''
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.backend as K
import os

def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, str)


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    """字符串转换为unicode格式（假设输入为utf-8格式）
    """

    if isinstance(text, bytes):
        text = text.decode(encoding, errors=errors)
    return text


def to_array(*args):
    """批量转numpy的array
    """
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results


def sequence_padding(inputs, length=None, padding=0, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        if mode == 'post':
            pad_width[0] = (0, length - len(x))
        elif mode == 'pre':
            pad_width[0] = (length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


def truncate_sequences(maxlen, index, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences


class ViterbiDecoder(object):
    """Viterbi解码算法基类
    """

    def __init__(self, trans, starts=None, ends=None):
        self.trans = trans
        self.num_labels = len(trans)
        self.non_starts = []
        self.non_ends = []
        if starts is not None:
            for i in range(self.num_labels):
                if i not in starts:
                    self.non_starts.append(i)
        if ends is not None:
            for i in range(self.num_labels):
                if i not in ends:
                    self.non_ends.append(i)

    def decode(self, nodes):
        """nodes.shape=[seq_len, num_labels]
        """
        # 预处理
        nodes[0, self.non_starts] -= np.inf
        nodes[-1, self.non_ends] -= np.inf

        # 动态规划
        labels = np.arange(self.num_labels).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        paths = labels
        for l in range(1, len(nodes)):
            M = scores + self.trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(0)
            scores = M.max(0).reshape((-1, 1))
            paths = np.concatenate([paths[:, idxs], labels], 0)

        # 最优路径
        return paths[:, scores[:, 0].argmax()]


class DataGenerator(object):
    """数据生成器模版
    """

    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def generate(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d

    def to_dataset(self, types, shapes, names=None, padded_batch=False):
        """转为tf.data.Dataset格式
        如果传入names的话，自动把数据包装成dict形式。
        """
        if names is None:
            generator = self.generate
        else:
            if is_string(names):
                def warps(k, v): return {k: v}
            elif is_string(names[0]):
                def warps(k, v): return dict(zip(k, v))
            else:
                def warps(k, v): return tuple(
                    dict(zip(i, j)) for i, j in zip(k, v)
                )

            def generator():
                for d in self.generate():
                    yield warps(names, d)

            types = warps(names, types)
            shapes = warps(names, shapes)

        if padded_batch:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types
            )
            dataset = dataset.padded_batch(self.batch_size, shapes)
        else:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types, output_shapes=shapes
            )
            dataset = dataset.batch(self.batch_size)

        return dataset


class NerDataGenerator(DataGenerator):

    def __init__(self, tokenizer, maxlen, class2id, data, batch_size, buffer_size=100):
        super().__init__(data, batch_size=batch_size, buffer_size=buffer_size)
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.class2id = class2id

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            token_ids, labels = [self.tokenizer._token_start_id], [0]
            for w, l in self.data[i]:
                w_token_ids = self.tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < self.maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = self.class2id[l] * 2 + 1
                        I = self.class2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    # 这部分还可以优化 TODO
                    break
            token_ids += [self.tokenizer._token_end_id]
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


class NamedEntityRecognizer(ViterbiDecoder):
    """
    命名实体识别器
    """

    def __init__(self, id2label, trans, starts, ends):
        super().__init__(trans, starts=starts, ends=ends)
        self.id2label = id2label

    def recognize(self, text, mapping, nodes):
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], self.id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


def ner_evaluate(NER: NamedEntityRecognizer, data, tokenizer, model):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        tokens = tokenizer.tokenize(text)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        R = set(NER.recognize(text, mapping, nodes))  # 预测
        T = set([tuple(i) for i in d if i[1] != 'O'])  # 真实
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    precision, recall = X / Y, X / Z
    f1 = 2*precision*recall/(precision+recall)
    return f1, precision, recall


class NerEvaluator(tf.keras.callbacks.Callback):
    def __init__(self,valid_data, CRF, NER, tokenizer, save_model_path):
        self.best_val_f1 = 0
        self.valid_data = valid_data
        self.CRF = CRF
        self.NER = NER
        self.save_model_path = save_model_path
        self.tokenizer = tokenizer

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(self.CRF.trans)
        self.NER.trans = trans
#         print(NER.trans)
        f1, precision, recall = ner_evaluate(self.NER, self.valid_data, self.tokenizer, self.model)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            print('Saved mode where f1 {}'.format(f1))
            p = os.path.join(self.save_model_path.format(epoch, f1))
            self.model.save_weights(p)
        print('Epoch-{}-valid:  f1: {:.5}, precision: {:.5}, recall: {:.5}, best f1: {:.5}\n'.format(epoch,f1, precision, recall, self.best_val_f1))