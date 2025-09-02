# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

import torch
import numpy
import pandas as pd
import json
emo_map=json.load(open('dic.json',encoding='utf-8'))
print(emo_map)
if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    dataset = 'dataset'  # 数据集
    model_name = 'bert'  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    config.device=torch.device('cpu')
    model = x.Model(config)
    model.load_state_dict(torch.load(config.save_path,map_location='cpu'))
    PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
    pad_size = 32
    print('加载完毕')
    print('输入文本：')
    content='大早晨的遇到一sb，把我车撞了，自己还挺有理。'
    print(content)
    token = config.tokenizer.tokenize(content)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)

    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    x = torch.LongTensor([token_ids]).to(config.device)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    text=(x,seq_len,mask)
    y=model(text)

    predc = torch.max(y.data, 1)[1].cpu().numpy()   #定义最大预测结果

    result = predc[0]
    print('预测结果:',emo_map[str(predc[0])])

