# https://qiita.com/TomokIshii/items/8d157211b8ab407707ce

#
#   linear_reg_chainer.py - Chainer version
#       date. 6/2/2017
#

import numpy as np

import chainer
from chainer import Function, Variable
import chainer.functions as F

# Target値 (3.0, 4.0), これを元に学習データサンプルを作成する．
W_target = np.array([[3.0]], dtype=np.float32)  # size = [1, 1]
b_target = 4.0

# Model Parameters
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
W = Variable(np.random.randn(1, 1).astype(np.float32) * 0.01,
             requires_grad=True)
b = Variable(np.zeros([1, 1], dtype=np.float32), requires_grad=True)


def model(x, W, b):
    # 線形回帰モデルの定義
    y1 = F.matmul(x, W)
    b1 = F.broadcast_to(b, x.shape)
    y = y1 + b1

    return y


def get_batch(W_target, b_target, batch_size=32):
    # バッチ・データの準備
    x = np.random.randn(batch_size, 1).astype(np.float32)
    y = x * W_target + b_target

    return Variable(x), Variable(y)


# Train loop
for batch_idx in range(100):
    # Get data
    batch_x, batch_y = get_batch(W_target, b_target)

    # Forward pass
    y_pred = model(batch_x, W, b)

    # 損失関数 MSE(mean square error)
    loss = F.mean_squared_error(y_pred, batch_y)

    # Manually zero the gradients after updating weights
    # パラメータの勾配をゼロ化する．（重要）
    W.cleargrad()
    b.cleargrad()

    # Backward pass
    loss.backward()

    # Apply gradients
    learning_rate = 0.1
    W.data = W.data - learning_rate * W.grad
    b.data = b.data - learning_rate * b.grad

    # Stop criterion
    if loss.data < 1.e-3:
        break

# 計算結果の出力
print('Loss: {:>8.4f} after {:d} batches'.format(
    float(loss.data), batch_idx))
print('==> Learned function:\t' + 'y = {:>8.4f} x + {:>8.4f}'.format(
    float(W.data), float(b.data)))
print('==> Actual function:\t' + 'y = {:>8.4f} x + {:>8.4f}'.format(
    float(W_target), float(b_target)))
