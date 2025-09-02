from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
from importlib import import_module
import numpy as np
import re

# 加载情感映射
emo_map = json.load(open('dic.json', encoding='utf-8'))

# 初始化模型
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

dataset = 'dataset'
model_name = 'bert'
x = import_module('models.' + model_name)
config = x.Config(dataset)
config.device = torch.device('cpu')

model = x.Model(config)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))
model.eval()

PAD, CLS = '[PAD]', '[CLS]'
pad_size = 32  # 设定固定的序列长度

# 创建 FastAPI 实例
app = FastAPI()


# 定义请求体格式
class TextRequest(BaseModel):
    text: str


# 数据预处理
def clean_weibo_text(text):
    # 移除@提及（使用正则）
    text = re.sub(r'@\S+', '', text)
    # 移除话题标签（#xxx#）
    text = re.sub(r'#.*?#', '', text).strip()
    # 移除URL
    text = re.sub(r'https?://\S+', '', text)
    # 特殊符号处理
    text = re.sub(r'[《》【】()（）]', '', text)
    # 空格合并
    text = re.sub(r'\s+', ' ', text).strip()
    # 去除多个重复符号
    pattern = r'([^\u4e00-\u9fa50-9a-zA-Z])\1+'
    text = re.sub(pattern, r'\1', text)
    pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9\u3000-\u303f\uff00-\uffef]')
    text = ''.join(pattern.findall(text))

    return text


@app.post("/predict")
def predict(request: TextRequest):
    content = request.text
    content = clean_weibo_text(content)
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
    text_input = (x, seq_len, mask)

    with torch.no_grad():
        y = model(text_input)
        predc = torch.max(y.data, 1)[1].cpu().numpy()
        result = predc[0]
        label = emo_map[str(predc[0])]
        print('预测结果:', emo_map[str(predc[0])])

    return {"prediction": label}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
