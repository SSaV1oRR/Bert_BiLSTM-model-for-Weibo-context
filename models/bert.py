# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf_8_sig').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练 1000
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 5                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小 128
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切) 32
        self.learning_rate = 5e-5                                       # 学习率
        # self.learning_rate = 1e-4
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=100,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # print(pooled.shape)

        data=pooled.view(1, pooled.shape[0], pooled.shape[1])
        # print(data,data.shape)
        lstm_out, _ = self.lstm(data)
        out = self.fc(pooled)
        # print('bert')
        return out





class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf_8_sig').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 5
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 2e-5

        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_hidden_size = 768
        self.lstm_hidden_size = 100
        self.hidden_size = self.lstm_hidden_size * 2  # BiLSTM 输出为双向


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=config.bert_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(config.hidden_size, 1)  # 注意力层
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # token_ids
        mask = x[2]     # attention_mask

        # BERT 编码所有 token
        sequence_output, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)

        # 送入 BiLSTM
        lstm_out, _ = self.lstm(sequence_output)  # [batch_size, seq_len, hidden_size*2]

        # Attention pooling
        attn_weights = self.attention(lstm_out)         # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # 按 seq_len 归一化
        out = torch.sum(attn_weights * lstm_out, dim=1)    # 加权求和 -> [batch_size, hidden_size*2]

        out = self.dropout(out)
        out = self.fc(out)
        return out
