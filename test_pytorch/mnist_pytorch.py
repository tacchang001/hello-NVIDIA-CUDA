# https://qiita.com/hagi-suke/items/e4b0310ebfb73b12313e

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1.データセットを用意する
training_epochs = 5            # エポック数
batch_size = 10                # バッチサイズ

trainset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True)

testset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size,
                                            shuffle=False)

# 2.モデルの作成
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(784,64)
        self.l2 = nn.Linear(64, 10)

    def forward(self, x):
        h = x.view(-1, 28 * 28) # 28*28の画像から１次元ベクトルにリサイズする
        h = F.relu(self.l1(h))
        return self.l2(h)

# 3.実行する
# GPUに対応させる
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

# 誤差逆伝播法アルゴリズムを選択する
criterion = nn.CrossEntropyLoss() # 損失関数を選択
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練する
for epoch in range(training_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # 計算前に勾配をゼロにする
        outputs = model(inputs) # 順伝播の計算をする
        loss = criterion(outputs, labels) # 損失を計算する
        loss.backward() # 誤差逆伝播の計算をする
        optimizer.step() # 更新する

        # 損失を定期的に出力する
        running_loss += loss.item()
        if i % 100 == 99:
            print('[{:d}, {:5d}] loss: {:3f}'
                    .format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# テストする
correct = 0
total = 0

with torch.no_grad():
    for (images, labels) in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))
