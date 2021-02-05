#! -*- coding: utf-8 -*-
# desc:
import json
from bert4one.snippets import is_string
from bert4one.models import MODELS
from loguru import logger


def build_bert_model(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    return_keras_model=True,
    **kwargs
):
    """
        config_path: 根据配置文件构建模型
        checkpoint_path: 可选的加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)

    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob', 512)
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)

    if is_string(model):
        model = model.lower()
        if model in MODELS:
            MODEL = MODELS[model]
        else:
            raise ValueError(f'Oops, model {model} not implemented')
    else:
        MODEL = model

    bert = MODEL(**configs)
    bert.build(**configs)

    if checkpoint_path is not None:
        logger.info(f"load checkpoint from {checkpoint_path}")
        bert.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return bert.model
    else:
        return bert