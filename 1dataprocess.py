import  pandas as pd
import json
df=pd.read_csv('data.csv')
emos=df['label'].values.tolist()
train_texts=df['text'].values.tolist()
emos_list=list(set(emos))
emos_map={}
i=0
while i<len(emos_list):
    emos_map[i]=emos_list[i]
    emos_map[emos_list[i]]=i
    i=i+1
#保存情感字典映射
with open("dic.json", "w", encoding="utf-8") as f:
    json.dump(emos_map, f, ensure_ascii=False, indent=4)

#生成训练数据
file1 = open('dataset/data/train.txt', 'w+', encoding='utf-8')
i = 0
while i < int(len(train_texts)*0.7):
    file1.write(str(train_texts[i]).strip().replace('\t','')+'\t'+str(emos_map[emos[i]]))
    file1.write('\n')
    i=i+1

#生成验证集和测试集
file2 = open('dataset/data/test.txt', 'w+', encoding='utf-8')
i = int(len(train_texts)*0.7)
while i < int(len(train_texts)*0.85):
    file2.write(str(train_texts[i]).strip().replace('\t','')+'\t'+str(emos_map[emos[i]]))
    file2.write('\n')
    i=i+1
file2 = open('dataset/data/dev.txt', 'w+', encoding='utf-8')
i = int(len(train_texts)*0.85)
while i < len(train_texts):
    file2.write(str(train_texts[i]).strip().replace('\t','')+'\t'+str(emos_map[emos[i]]))
    file2.write('\n')
    i=i+1
#class数据
i=0
file=open('dataset/data/class.txt','w+',encoding='utf-8')
while i<len(emos_list):
    file.write(str(i))
    file.write('\n')
    i=i+1
file.close()
print('done!')