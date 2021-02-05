import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.python.util import nest, tf_inspect
from tensorflow.python.eager import tape
from distutils.util import strtobool
import os

# 判断是否启用重计算（通过时间换空间）
do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))


def sequence_masking(x, mask, mode=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    mode: 如果是0，则直接乘以mask；
          如果是1，则在padding部分减去一个大正数。
    axis: 序列所在轴，默认为1；
    """
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if axis is None:
            axis = 1
        if axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12


def recompute_grad(call):
    """重计算装饰器（用来装饰Keras层的call函数）
    关于重计算，请参考：https://arxiv.org/abs/1604.06174
    """
    if not do_recompute:
        return call

    def inner(self, inputs, **kwargs):
        """定义需要求梯度的函数以及重新定义求梯度过程
        （参考自官方自带的tf.recompute_grad函数）
        """
        flat_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            """定义前向计算
            """
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            """定义前向计算和反向计算
            """
            with tape.stop_recording():
                outputs = kernel_call()
                outputs = tf.identity(outputs)

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(
                    outputs, watches, output_gradients=[doutputs]
                )
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        outputs, grad_fn = call_and_grad(*flat_inputs)
        flat_outputs = nest.flatten(outputs)

        def actual_grad_fn(*doutputs):
            grads = grad_fn(*doutputs, variables=self.trainable_weights)
            return grads[0] + grads[1]

        watches = flat_inputs + self.trainable_weights
        watches = [tf.convert_to_tensor(x) for x in watches]
        tape.record_operation(
            call.__name__, flat_outputs, watches, actual_grad_fn
        )
        return outputs

    return inner

class Sinusoidal(keras.initializers.Initializer):
    """Sin-Cos位置向量初始化器
    来自：https://arxiv.org/abs/1706.03762
    """
    def __call__(self, shape, dtype=None):
        """Sin-Cos形式的位置向量
        """
        vocab_size, depth = shape
        embeddings = np.zeros(shape)
        for pos in range(vocab_size):
            for i in range(depth // 2):
                theta = pos / np.power(10000, 2. * i / depth)
                embeddings[pos, 2 * i] = np.sin(theta)
                embeddings[pos, 2 * i + 1] = np.cos(theta)
        return embeddings

keras.utils.get_custom_objects().update({
    'Sinusoidal': Sinusoidal,
})