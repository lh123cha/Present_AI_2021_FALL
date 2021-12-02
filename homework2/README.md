# Homework2

姓名：梁辉

学号：10185501411

## 网络结构

网络包含四层卷积层，2层全连接层。

```python
ConvNet(
  (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
  (relu1): ReLU()
  (batch1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1))
  (relu2): ReLU()
  (batch2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv1_drop): Dropout(p=0.25, inplace=False)
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (relu3): ReLU()
  (batch3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  (relu4): ReLU()
  (batch4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2_drop): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=576, out_features=256, bias=True)
  (fc1_relu): ReLU()
  (dp1): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=256, out_features=10, bias=True)
  (softmax): log_softmax(dim=1)
)
```

**图示化网络结构**

![](E:\.idea\大四上\当代人工智能\homework2\网络结构.png)



## 实验结果

首先将数据分为训练集，验证集和测试集，验证集用来调整参数，不在测试集上调整。测试集只用来做最后的测试准确率。

先在训练集上训练，在测试集上调整参数，在验证集上保存损失最小的模型，并判断损失是否震荡，采用early stoping结束训练。

训练20个epoch之后，在验证集上的准确率$99.5%$。

![](E:\.idea\大四上\当代人工智能\homework2\准确率.PNG)

将模型使用在预测测试集上，得到测试集上准确率$99\%$

![](E:\.idea\大四上\当代人工智能\homework2\准确率结果2.png)

## 实验总结

通过本次实验，我认识到了之前那种将没跑一遍训练集在测试集上测试一下准确率的做法是大错特错的，这次实验将这种错误及时改正，对于数据集的划分最好采用交叉验证进行划分，但这次由于时间仓促，只是固定了训练集和验证集，在验证集上验证好准确率的模型保存下来，在测试集上预测得到最终的结果。

之后的实验会从这次实验中吸取教训，仔细认真的处理数据，构建模型。
