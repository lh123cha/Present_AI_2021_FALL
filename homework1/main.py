# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import logging
import numpy as np
import os
import sys
from torch.utils.data import DataLoader,Dataset
import json
import gensim
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)

#Get the network config
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()
with open(args.config_path, 'r',encoding='utf-8') as f:
    args = json.load(f)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

train_data=torch.load('./data/trainTensor0-4.pt')
train_labels=torch.load('./data/trainLabels0-4.pt')
test_data=torch.load('./data/testTensor0-4.pt')
test_labels=torch.load('./data/testLabels0-4.pt')
train_labels=train_labels.long()
test_labels=test_labels.long()

all_correct_dict=[]
all_loss_dict=[]
correct_dict=[]
loss_dict=[]

lrs=[0.001,0.005,0.01,0.05,0.1]

#define the dataset
class CustomTextDataset(Dataset):
    def __init__(self,text_tensor,image_labels,transform=None,target_transform=None):
        self.img_labels=image_labels
        self.transform=transform
        self.target_transorm=target_transform
        self.text_tensor=text_tensor
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        text=self.text_tensor[idx]
        label=self.img_labels[idx]
        if self.transform:
            text=self.transform(text)
        if self.target_transorm:
            label=self.target_transorm(label)
        return text,label
# Define model
# Use LSTM Modul
class TextRnnModel(nn.Module):
    def __init__(self, embedding_size, lstm_hidden_size, output_size, num_layers=2, dropout=0.3):
        super(TextRnnModel, self).__init__()
        self.lstm = nn.LSTM(embedding_size,  # 词嵌入模型词语维度
                            lstm_hidden_size,  # 隐层神经元的维度，为输出的维度
                            num_layers,  # 构建两层的LSTM：堆叠LSTM
                            bidirectional=True,  # 双向的LSTM：词向量从前往后走，再重后往前走，最后拼接起来
                            batch_first=True,  # 把第一个维度的输入作为batch输入的维度
                            dropout=dropout)
        self.fc = nn.Linear(lstm_hidden_size * 2, output_size)
        # nn.Linear(输入维度，输出维度)：[ 上一步输出的LSTM维度*2(双向) , 10分类 ]

    def forward(self, x):
        """前向传播"""
        out, _ = self.lstm(x)  # 过一个LSTM [batch_size, seq_len, embedding]
        out = self.fc(out[:, -1, :])
        return out

# Use MLP Modul
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(150*10, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1500),
            nn.ReLU(),
            nn.Linear(1500,1500),
            nn.ReLU(),
            nn.Linear(1500, int(args['num_labels']))
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        size = len(dataloader.dataset)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    correct_dict.append(100*correct)
    loss_dict.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#tramform the data into pytorch class
training_data=CustomTextDataset(train_data,train_labels)
testing_data=CustomTextDataset(test_data,test_labels)
epochs = args['num_epochs']
if args['type']=="rnn":
    batch_size = 128
    embedding_size = 10
    output_class = 5
    # print(x_trainVB          [0][0])
    train_loader = DataLoader(training_data,batch_size=128,shuffle=True)#使用DataLoader加载数据
    test_loader = DataLoader(testing_data,batch_size=128,shuffle=True)
    model = TextRnnModel(embedding_size=embedding_size, lstm_hidden_size=200, output_size=output_class)
    cross_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # 优化器
    model.train()
    for i in range(epochs):
        train(train_loader,model,cross_loss,optimizer)
        test(test_loader,model,cross_loss)
    torch.save(model, './data/lstm_rnn.pkl')
    print("Done!")
if args['type']=="mlp":
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)  # 使用DataLoader加载数据
    test_dataloader = DataLoader(testing_data, batch_size=128, shuffle=True)

    model = NeuralNetwork().to(device)
    model=model.double()
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    for lr in lrs:
        if lr==0.05:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            epochs = 150
            epochs_dict=[]
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                train(train_dataloader, model, loss_fn, optimizer)
                test(test_dataloader, model, loss_fn)
                epochs_dict.append(t)
            all_correct_dict.append(correct_dict)
            all_loss_dict.append(loss_dict)
            correct_dict=[]
            loss_dict=[]
        else:
            pass
    plt.plot(epochs_dict,all_correct_dict[0],color='C0',label='acc-0.001')
    # plt.plot(epochs_dict, all_correct_dict[1], color='C2', label='acc-0.005')
    # plt.plot(epochs_dict, all_correct_dict[2], color='C4', label='acc-0.01')
    # plt.plot(epochs_dict, all_correct_dict[3], color='C6', label='acc-0.05')
    # plt.plot(epochs_dict, all_correct_dict[4], color='C8', label='acc-0.1')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy %')
    plt.title("Accuracy chart")
    plt.legend()
    plt.savefig('./doc/all_mlp_acc.jpg')  # 保存图片，路径名为test.jpg
    plt.show()


    plt.plot(epochs_dict, all_loss_dict[0], color='C1', label='loss-0.001')
    # plt.plot(epochs_dict, all_loss_dict[1], color='C3', label='loss-0.005')
    # plt.plot(epochs_dict, all_loss_dict[2], color='C5', label='loss-0.01')
    # plt.plot(epochs_dict, all_loss_dict[3], color='C7', label='loss-0.05')
    # plt.plot(epochs_dict, all_loss_dict[4], color='C9', label='loss-0.1')
    plt.xlabel('epochs')
    plt.ylabel('loss function')
    plt.title("loss chart")
    plt.legend()
    plt.savefig('./doc/all_mlp_loss.jpg')  #保存图片，路径名为test.jpg
    plt.show()               #显示图片
    torch.save(model, './data/2layer_mlp.pkl')

print("Done!")




