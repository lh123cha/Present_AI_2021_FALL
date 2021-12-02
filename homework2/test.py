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
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
test_db=datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=BATCH_SIZE, shuffle=True)
PATH='./minist_model.pkl'
model_dict=torch.load(PATH)
test_loss,test_acc=test(test_loader,model_dict)
print('Test_loss:%.5f , Test_acc:%.5f' % (
    test_loss, test_acc))
