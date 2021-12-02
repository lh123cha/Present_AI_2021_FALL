#Homework2

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
BATCH_SIZE=128

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

training=False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
# 训练集
train_db=datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),       # 图像转化为Tensor
                           transforms.Normalize((0.1307,), (0.3081,))       # 标准化
                       ]))
test_db=datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

train_db, val_db = torch.utils.data.random_split(train_db, [55000, 5000])
# 测试集
test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=BATCH_SIZE, shuffle=True)            # shuffle() 方法将序列的所有元素随机排序
#训练集和验证集
train_loader = torch.utils.data.DataLoader(                 # vision.utils : 用于把形似 (3 x H x W) 的张量保存到硬盘中，给一个mini-batch的图像可以产生一个图像格网。
        train_db,
        batch_size=BATCH_SIZE, shuffle=True)            # shuffle() 方法将序列的所有元素随机排序
val_loader = torch.utils.data.DataLoader(
      val_db,
      batch_size=BATCH_SIZE, shuffle=True)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)#norm2d归一化

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_drop = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_drop = nn.Dropout(0.25)

        # Fully-Connected layer 1

        self.fc1 = nn.Linear(576, 256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batch1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)

        out = self.maxpool1(out)
        out = self.conv1_drop(out)

        # conv layer 2 的前向计算，4行代码
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batch3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.batch4(out)

        out = self.maxpool2(out)
        out = self.conv2_drop(out)

        # Flatten将卷积得到的张量拉平接入全连接网络
        out = out.view(out.size(0), -1)

        # FC layer的前向计算
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.dp1(out)

        out = self.fc2(out)

        return F.log_softmax(out, dim=1)



def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = []
    train_acc = []

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data

        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)

        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum().item()
        acc = num_correct / len(labels)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_acc.append(acc)

    return np.mean(train_loss), np.mean(train_acc)


def test(test_loader, model):
    model.eval()
    test_loss = []
    test_acc = []

    for i, data in enumerate(test_loader, 0):

        inputs, labels = data

        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum().item()
        acc = num_correct / len(labels)

        test_loss.append(loss.item())
        test_acc.append(acc)

    return np.mean(test_loss), np.mean(test_acc)


# def get_k_fold_data(k, i, X, y):
#     # 返回第i折交叉验证时所需要的训练和测试数据，分开放，X_train为训练数据，X_test为验证数据
#     assert k > 1
#     fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（向下取整）
#
#     X_train, y_train = None, None
#     for j in range(k):
#         idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数 得到测试集的索引
#
#         X_part, y_part = X[idx, :], y[idx]  # 只对第一维切片即可
#         if j == i:  # 第i折作test
#             X_test, y_test = X_part, y_part
#         elif X_train is None:
#             X_train, y_train = X_part, y_part
#         else:
#             X_train = torch.cat((X_train, X_part), dim=0)  # 其他剩余折进行拼接 也仅第一维
#             y_train = torch.cat((y_train, y_part), dim=0)
#     return X_train, y_train, X_test, y_test
# X_train,y_train,
if training:
    model = ConvNet()
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=30, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=20, min_lr=0.0001, eps=1e-08)
    if use_cuda:
        model = model.cuda()

    best_loss = 100000

    for epoch in range(0, 15):
        time_all = 0
        start_time = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        time_all = time.time() - start_time
        # scheduler.step(test_loss)
        val_loss,val_acc=test(val_loader,model)
        print('Epoch: %d,  Train_loss: %.5f, Train_acc: %.5f , T_Time: %.3f , Val_loss: %.5f, Val_acc: %.5f' % (
        epoch, train_loss, train_acc, time_all,val_loss,val_acc))
        if(val_loss<best_loss):
            best_loss=val_loss
            torch.save(model, 'minist_model.pkl')
            print("already save the model")
        if(val_loss-best_loss>0.01 and epoch>1):#early stoping如果验证损失过大则提前终止。
            break
else:
    criterion = nn.CrossEntropyLoss()
    model_dict=torch.load('minist_model.pkl')
    test_loss,test_acc=test(test_loader,model_dict)
    print('Test_loss: %.5f , Test_acc: %.5f'%(test_loss,test_acc))
