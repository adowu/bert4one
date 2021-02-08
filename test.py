# -*- encoding: utf-8 -*-
'''
@Email :  wushaojun0107@gmail.com
@Time  :  02-07 2021
'''

import tensorflow as tf
import tensorflow.keras.backend as K


y_pred = K.constant([[[0.4, 0.6], [0.3, 0.7]], [[0.2, 0.8], [0.4, 0.6]]])
print(y_pred)
print('----------pred')
print(y_pred[:, 0])

y_true = K.constant([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
print(y_true)
#  逐个位置 相乘 然后相加
a = tf.einsum('bni,bni->b', y_true, y_pred)
print('aaaaaaaaaa')
print(a)

trans = K.constant([[1, 3], [2, 4]])
print('trans')
print(trans)
x = y_true[:, :-1]
y = y_true[:, 1:]
print('xxxxxxxxx')
print(x)
print('yyyyyyyy')
print(y)
b = tf.einsum('bni, ij, bnj -> b', x, trans, y)
print('bbbbbbb')
print(b)


a = K.random_normal(shape=[1, 2, 1])
b = K.random_normal(shape=[1, 2, 2])
print(a)
print(b)
c = a+b
d = K.sum(c, 1)
print(c)

print(d)


a = K.random_normal(shape=[2, 1])
b = K.random_normal(shape=[2, 2, 2])
print(a)
print(b)
c = a+b
d = K.sum(c, 1)
print(c)
print('------------------------------------')
y_pred = K.constant([[[0.4, 0.6], [0.3, 0.7]], [[0.2, 0.8], [0.4, 0.6]]])
print(y_pred)
mask = K.all(K.greater(y_pred, -1e6), axis=2, keepdims=True)
print(mask)
mask = K.cast(mask, K.floatx())
print(mask)


print('__________________________')

a = K.random_normal(shape=[2, 2, 2])
b = K.random_normal(shape=[2, 2, 1])

c = K.concatenate([a, b], axis=2)
print(c)

d = b[:,:1]
print(b)
print(d)