#使用word2vec将文本转化为向量
import json
import gensim
import numpy as np
from tensorflow import keras
import os
from tensorflow.keras.preprocessing import sequence
import argparse
import logging
import os
import random
import shutil
from shutil import copy2
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()
with open(args.config_path, 'r',encoding='utf-8') as f:
    args = json.load(f)

def load_file():
    labels=[]
    json_data=[]
    type1_data=[]

    #加载停用词表
    stop_words_file_path="E:/.idea/大四上/当代人工智能/hw1/data/停用词.txt"
    stop_words_file = open(stop_words_file_path, 'r',encoding="utf-8",)
    stop_words = list()
    for line in stop_words_file.readlines():
        line = line.strip()   # 去掉每行末尾的换行符
        stop_words.append(line)

    file_path="E:/.idea/大四上/当代人工智能/hw1/data/20news.json"
    with open(file_path,'r',encoding='utf8')as fp:
        for line in fp.readlines():
            dic = json.loads(line)
            json_data.append(dic)
    all_word=[]
    #对前五种文章进行五分类
    #采用五分类方法或者直接20分类
    use_five_split=args['use_5split']
    data_length=0
    if use_five_split:
        data_length=3764
    else:
        data_length=len(json_data)
    for i in range(14062,18827):
        labels.append(json_data[i]['label'])
        single_word=[]
        for word in json_data[i]['text']:
            if(word in stop_words):
                pass
            else:
                single_word.append(word)
        all_word.append(single_word)
    return all_word, labels # 总文本，总标签
## 3.获取word2vec模型， 并构造，词语index字典，词向量字典
def get_word2vec_dictionaries(texts):
    def get_word2vec_model(texts=None): # 获取 预训练的词向量 模型，如果没有就重新训练一个。
        if os.path.exists('./20news_word2vec_10'): # 如果训练好了 就加载一下不用反复训练
            model = gensim.models.Word2Vec.load('./20news_word2vec_10')
            # print(model['作者'])
            return model
        else:
            model = gensim.models.Word2Vec(texts, vector_size =int(args['embedding_len']), window=7, min_count=10, workers=4,sg=1)
            model.save('./20news_word2vec_10') # 保存模型
            return model

    Word2VecModel = get_word2vec_model(texts) #  获取 预训练的词向量 模型，如果没有就重新训练一个。

    vocab_list = Word2VecModel.wv.index_to_key  # 存储 所有的 词语
    print(len(vocab_list))

    word_index = {" ": 0}# 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    word_vector = {} # 初始化`[word : vector]`字典

    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

    ## 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵

    return word_index, word_vector, embeddings_matrix
# 序号化 文本，tokenizer句子，并返回每个句子所对应的词语索引


def tokenizer(texts, word_index):

    data = []
    for sentence in texts:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(word_index[word])  # 把句子中的 词语转化为index
            except:
                new_txt.append(0)

        data.append(new_txt)

    res_texts = sequence.pad_sequences(data, maxlen = int(args['max_seqence_length']))  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    return res_texts

def get_text_and_lable_together(data,label):
    res=[]
    for i in range(len(label)):
        _tuple=(data[i],label[i])
        res.append(_tuple)
    return res

def data_set_split(data):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param data 全部数据
    :return:
    '''
    train_size = int(float(args['validation_split']) * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    return train_dataset,test_dataset

texts,labels=load_file()
word_inedx,word_vector,embedding_matrix=get_word2vec_dictionaries(texts)
res=tokenizer(texts,word_inedx)
text_and_lable=get_text_and_lable_together(res,labels)
train_data,test_data=data_set_split(text_and_lable)

train_load=[]
train_labels=[]
test_load=[]
test_labels=[]
for x in range(len(train_data)):
    temp=[]
    train_labels.append(train_data[x][1])
    for i in train_data[x][0]:
        temp.append(embedding_matrix[i])
    train_load.append(temp)
train_load=np.array(train_load)
train_labels=np.array(train_labels)
train_load=torch.from_numpy(train_load)
train_labels=torch.from_numpy(train_labels)
train_load.to(torch.float32)
train_labels.to(torch.float32)
print("tarin_data is: ")
print(train_load)
print(train_labels)

for x in range(len(test_data)):
    temp=[]
    test_labels.append(test_data[x][1])
    for i in test_data[x][0]:
        temp.append(embedding_matrix[i])
    test_load.append(temp)
test_labels=np.array(test_labels)
test_load=np.array(test_load)
test_load=torch.from_numpy(test_load)
test_labels=torch.from_numpy(test_labels)
test_load.to(torch.float32)
test_labels.to(torch.float32)
print("test_data is : ")
print(test_load)
print(test_labels)

torch.save(train_load, "./data/all_trainTensor15-19.pt")
torch.save(train_labels,'./data/all_trainLabels15-19.pt')
torch.save(test_load,'./data/all_testTensor15-19.pt')
torch.save(test_labels,'./data/all_testLabels15-19.pt')
