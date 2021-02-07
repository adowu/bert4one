# -*- encoding: utf-8 -*-
'''
@Email :  wushaojun0107@gmail.com
@Time  :  02-05 2021
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda
import tensorflow.keras.backend as K


def search_layer(inputs, name, exclude=None):
    """
        根据inputs和name来搜索层
    """
    if exclude is None:
        exclude = set()

    if isinstance(inputs, Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]
    if layer.name == name:
        return layer
    elif layer in exclude:
        return None
    else:
        exclude.add(layer)
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                print('layer: ', layer)
                layer = search_layer(layer, name, exclude)
                if layer is not None:
                    return layer


class MaximumEntropyMarkovModel(Layer):
    
    def __init__(self, lr_multiplier=1, **kwargs):
        super(MaximumEntropyMarkovModel, self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier  # 当前层学习率的放大倍数

    def build(self, input_shape):
        output_dim = input_shape[-1]
        if not isinstance(output_dim, int):
            output_dim = output_dim.value
        self.trans = self.add_weight(name='trans',
                                     shape=(output_dim, output_dim),
                                     initializer='glorot_uniform',
                                     trainable=True)
        if self.lr_multiplier != 1:
            K.set_value(self.trans, K.eval(self.trans) / self.lr_multiplier)
            self.trans = self.lr_multiplier * self.trans

    def call(self, inputs, mask=None):
        """MEMM本身不改变输出，它只是一个loss
        """
        if mask is not None:
            if not hasattr(self, 'mask_layer'):
                self.mask_layer = search_layer(inputs, mask)

        return inputs

    @property
    def output_mask(self):
        if hasattr(self, 'mask_layer'):
            return self.mask_layer.output_mask

    def reverse_sequence(self, inputs, mask=None):
        if mask is None:
            return [x[:, ::-1] for x in inputs]
        else:
            length = K.cast(K.sum(mask, 1), 'int32')
            return [
                tf.reverse_sequence(x, length, seq_axis=1)
                for x in inputs
            ]

    def basic_loss(self, y_true, y_pred, go_backwards=False):
        """y_true需要是整数形式（非one hot）"""
        mask = self.output_mask
        y_true = K.cast(y_true, 'int32')
        y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])

        if go_backwards:
            y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
            trans = K.transpose(self.trans)
        else:
            trans = self.trans

        # loss
        history = K.gather(trans, y_true)
        history = K.concatenate([y_pred[:, :1], history[:, :-1]], 1)
        y_pred = (y_pred + history) / 2
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True)
        if mask is None:
            return K.mean(loss)

        else:
            return K.sum(loss*mask) / K.sum(mask)

    def sparse_loss(self, y_true, y_pred):
        """y_true需要是整数形式（非one hot）"""
        loss = self.basic_loss(y_true, y_pred, False)
        loss += self.basic_loss(y_true, y_pred, True)
        return loss / 2

    def basic_accuracy(self, y_true, y_pred, go_backwards=False):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        mask = self.output_mask
        # y_true需要重新明确一下dtype和shape
        y_true = K.cast(y_true, 'int32')
        y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
        # 是否反转序列
        if go_backwards:
            y_true, y_pred = self.reverse_sequence([y_true, y_pred], mask)
            trans = K.transpose(self.trans)
        else:
            trans = self.trans
        # 计算逐标签accuracy
        histoty = K.gather(trans, y_true)
        histoty = K.concatenate([y_pred[:, :1], histoty[:, :-1]], 1)
        y_pred = (y_pred + histoty) / 2
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        isequal = K.cast(K.equal(y_true, y_pred), K.floatx())
        if mask is None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)

    def sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        accuracy = self.basic_accuracy(y_true, y_pred, False)
        accuracy = accuracy + self.basic_accuracy(y_true, y_pred, True)
        return accuracy / 2

    def dense_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_accuracy(y_true, y_pred)

    def get_config(self):
        config = {
            'lr_multiplier': self.lr_multiplier,
        }
        base_config = super(MaximumEntropyMarkovModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
