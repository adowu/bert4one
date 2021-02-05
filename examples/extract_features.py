# -*- encoding: utf-8 -*-
'''
@Email :  wushaojun0107@gmail.com
@Time  :  02-04 2021
'''
import os
import shutil
import sys
# 修改执行目录
sys.path[0] = os.getcwd()
import tensorflow.keras as keras
import numpy as np
from loguru import logger
from bert4one.snippets import to_array
from bert4one.tokenizer import Tokenizer
from bert4one.build import build_bert_model

root_path = '/Users/shaojun7/models'
sub_path = ['chinese_roberta_wwm_base','chinese_google_base', 'nezha_base_wwm', ]
model = ['robert', 'bert', 'nezha']
index = 2


config_path = os.path.join(root_path, sub_path[index], 'bert_config.json')
checkpoint_path = os.path.join(root_path, sub_path[index], 'bert_model.ckpt')
vocab_path = os.path.join(root_path, sub_path[index], 'vocab.txt')

tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_bert_model(config_path, checkpoint_path, model[index])

token_ids, segment_ids = tokenizer.encode('好好学习别玩游戏了好吧')
token_ids, segment_ids = to_array([token_ids], [segment_ids])

results = model.predict([token_ids, segment_ids])
logger.debug(np.shape(results))
logger.debug(results)
logger.debug('\n ===== storing =====\n')
model.save('test.model')
del model
logger.debug('\n ===== reloading =====\n')
model = keras.models.load_model(filepath='test.model')
logger.debug(model.predict([token_ids, segment_ids]))
shutil.rmtree('test.model')
