"""
session2のメモ及びコード

* PyTorch
Define by Run
Numpyと類似したTensorクラス
"""


# pytorchによる手書き文字分類のサンプル
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

mnist_train = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

print(f"学習データ数: {len(mnist_train)}, 評価データ数: {len(mnist_test)}")

# Dataloderの設定
img_size = 28
batch_size = 256
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# モデルの構築
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(img_size * img_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forword(self, x):
        x = x.view(-1, img_size * img_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


img_clf = ImageClassifier()
print(img_clf)

# 学習
from torch import optim

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(img_clf.parameters(), lr=0.01)

record_loss_train = []
record_loss_test = []

epoch = 5

for i in range(epoch):
    img_clf.train()
    loss_train = 0

    # ミニバッチ
    for j, (x, y) in enumerate(train_loader):
        pred = img_clf.forword(x)
        loss = loss_fn(pred, y)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train /= j + 1
    record_loss_train.append(loss_train)

    img_clf.eval()
    loss_test = 0
    for j, (x, y) in enumerate(test_loader):
        pred = img_clf.forword(x)
        loss = loss_fn(pred, y)
        # print(f"test j: {j}, loss_item: {loss.item()}")
        loss_test += loss.item()
    loss_test /= j + 1
    record_loss_test.append(loss_test)

    if i % 1 == 0:
        print(f"Epoch: {i}, LossTrain: {loss_train}, LossTest: {loss_test}")

# 誤差可視化
import matplotlib.pyplot as plt

plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.show()