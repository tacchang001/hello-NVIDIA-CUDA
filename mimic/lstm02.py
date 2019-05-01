# https://helve-python.hatenablog.jp/entry/2018/11/22/000000

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, GRU, LSTM

# 時系列データの生成と変換

timesteps = 5

x_base = np.array([-1,-1,0,0,1,1,0,0], dtype=np.float32)
x = np.empty(0, dtype=np.float32)

for i in range(1000):
    x = np.hstack([x, x_base])

xdata = np.array([x[i:i+timesteps] for i in range(len(x)-timesteps)])
xdata = xdata.reshape(xdata.shape[0], timesteps, -1)

ydata = x[timesteps:].reshape(xdata.shape[0], -1)


# モデル作成

actfunc = "tanh"

model = Sequential()
# model.add(SimpleRNN(10, activation=actfunc,
#                     batch_input_shape=(None, timesteps, 1)))
# model.add(GRU(10, activation=actfunc,
#               batch_input_shape=(None, timesteps, 1)))
model.add(LSTM(10, activation=actfunc,
               batch_input_shape=(None, timesteps, 1)))
model.add(Dense(10, activation=actfunc))
model.add(Dense(1))

# 学習の実行

model.compile(optimizer='sgd',
              loss='mean_squared_error')

history = model.fit(xdata, ydata,
                    batch_size=100,
                    epochs=150,
                    verbose=1)

# 予測値を出力

pred = model.predict(xdata)

fig, ax = plt.subplots()
ax.plot(ydata[:20, :].reshape(-1), linewidth=0, marker="o", markersize=8)
ax.plot(pred[:20, :].reshape(-1), linewidth=0, marker="o", markersize=5)
ax.set_xticks(np.arange(0, 20, 2))
ax.set_yticks([-1, 0, 1])
ax.legend(["training", "prediction"])
ax.grid()
plt.show()

# 損失関数をプロット

loss = history.history["loss"]

fig, ax = plt.subplots()
ax.plot(loss)
ax.legend(["LSTM"])
plt.show()

