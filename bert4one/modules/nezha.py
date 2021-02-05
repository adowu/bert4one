#! -*- coding: utf-8 -*-
# desc:
from bert4one.modules.bert import BERT
from tensorflow.keras.layers import Embedding, Add, Dropout, Dense
from bert4one.layers import LayerNormalization,MultiHeadAttention,FeedForward,RelativePositionEmbedding


class NEZHA(BERT):
    """华为推出的NAZHA模型
    链接：https://arxiv.org/abs/1909.00204
    """

    def apply_embeddings(self, inputs: list):
        """NEZHA 的embeddings 是 token 和 segment 两者 embedding之和"""
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        z = self.layer_norm_conds[0]
        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        if self.segment_vocab_size > 0:
            if self.shared_segment_embeddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=2,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )
        return x

    def apply_main_layers(self, inputs, index):
        """NEZHA的主体是基于Self-Attention的模块顺序：Att --> Add --> LN --> FFN --> Add --> LN"""
        x = inputs
        z = self.layer_norm_conds[0]
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi, x = x, [x, x, x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'typical_relative'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(3, attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )
        return x

    def compute_position_bias(self, inputs=None):
        """经典相对位置编码
        """
        if self.position_bias is None:

            x = inputs
            self.position_bias = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbedding,
                input_dim=2 * 64 + 1,
                output_dim=self.attention_head_size,
                embeddings_initializer='Sinusoidal',
                name='Embedding-Relative-Position',
                trainable=False
            )

        return self.position_bias