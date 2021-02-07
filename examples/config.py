# -*- encoding: utf-8 -*-
'''
@Email :  wushaojun0107@gmail.com
@Time  :  02-05 2021
'''
import os

class Config(object):

    def __init__(self) -> None:

        root_path = '/Users/shaojun7/models'
        sub_path = ['chinese_roberta_wwm_base','chinese_google_base', 'nezha_base_wwm', ]
        self.model = ['robert', 'bert', 'nezha']
        self.model_index = 1
        self.config_path = os.path.join(root_path, sub_path[self.model_index], 'bert_config.json')
        self.checkpoint_path = os.path.join(root_path, sub_path[self.model_index], 'bert_model.ckpt')
        self.vocab_path = os.path.join(root_path, sub_path[self.model_index], 'vocab.txt')

        self.save_model_path = 'save_model'
        # ner 语料
        self.china_daily_train = 'data/china-people-daily-ner-corpus/example.train'
        self.china_daily_dev = 'data/china-people-daily-ner-corpus/example.dev'
        self.china_daily_test = 'data/china-people-daily-ner-corpus/example.test'