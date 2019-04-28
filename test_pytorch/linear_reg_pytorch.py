# https://qiita.com/TomokIshii/items/8d157211b8ab407707ce

#
#   linear_reg_pytorch.py - PyTorch version
#       date. 5/24/2017
#

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Target値 (3.0, 4.0)
W_target = torch.FloatTensor([[3.0]])  # size = [1, 1]
b_target = 4.0

# Model Parameters
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
W = Variable((torch.randn(1, 1) * 0.01).type(dtype), requires_grad=True)
b = Variable(torch.zeros(1, 1).type(dtype), requires_grad=True)


def model(x):
    # 線形回帰モデルの定義
    y = torch.mm(x, W) + b.expand_as(x)
    return y


def get_batch(batch_size=32):
    # バッチ・データの準備
    x = torch.randn(batch_size, 1)
    y = torch.mm(x, W_target) + b_target
    return Variable(x), Variable(y)


# 損失関数 MSE(mean square error)
loss_fn = torch.nn.MSELoss(size_average=True)

# Train loop
for batch_idx in range(20):
    # Get data
    batch_x, batch_y = get_batch()

    # Forward pass
    y_pred = model(batch_x)
    loss = loss_fn(y_pred, batch_y)
    loss_np = loss.data[0]

    # Backward pass
    loss.backward()

    # Apply gradients
    learning_rate = 0.1
    W.data = W.data - learning_rate * W.grad.data
    b.data = b.data - learning_rate * b.grad.data

    # Manually zero the gradients by torch.Tensor.zero_()
    # パラメータの勾配をゼロ化する．（重要）
    W.grad.data.zero_()
    b.grad.data.zero_()

    # Stop criterion
    if loss_np < 1.e-3:
        break


# 計算結果の出力
def model_desc(W, b):
    # Support function to show result.
    if type(W) == torch.FloatTensor:
        W = W.numpy()
    if type(b) == torch.FloatTensor:
        b = b.numpy()
        b = float(b)

    result = 'y = {0} x + {1:>8.4f}'.format(W, b)

    return result


print('Loss: {:>8.4e} after {:d} batches'.format(loss_np, batch_idx))
print('==> Learned function:\t' + model_desc(W.data.view(-1), b.data))
print('==> Actual function:\t' + model_desc(W_target.view(-1), b_target))
