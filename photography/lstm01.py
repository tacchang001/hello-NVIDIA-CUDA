# https://mrsekut.site/?p=1019

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
from keras.utils.vis_utils import plot_model

# from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM

np.random.seed(0)


def NoiseCos(T=100, ampl=0.05, cycle=2):
    # amplifier:ノイズの程度
    # lowからhighまでのsize個の一様乱数を生成
    # cycle:描画するグラフの周期数
    t = np.arange(0, cycle * 2 * T + 1)
    return np.cos(2 * np.pi * t / T) + ampl * np.random.uniform(-1.0, 1.0, len(t))


# lossの履歴をプロット
def plot_history(history):
    plt.plot(history.history['loss'], label="", )
    plt.title('LSTM')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


###############################
#         データの生成          #
###############################
cycle = 2  # 描画する周期の個数
T = 100  # 周期
f = NoiseCos(T)  # f[t]でアクセス

TrainDataLength = 2 * cycle * T  # 全時系列の長さ
maxlen = 50  # 1つの時系列データの長さ(前回maxlen分のデータを次回に適用

InputData = []  # 訓練データ
TeacherData = []  # 教師データ

for i in range(0, TrainDataLength - maxlen + 1):
    InputData.append(f[i: maxlen + i])
    TeacherData.append(f[i + maxlen])

X = np.array(InputData).reshape(len(InputData), maxlen, 1)  # 訓練データ
Y = np.array(TeacherData).reshape(len(TeacherData), 1)  # 教師データ
# print(X.shape)  # 151,50,1
# print(Y.shape)  # 151, 1

# 訓練データ・検証データに分割(N_train:N_validation = 9:1)
N_train = int(len(InputData) * 0.9)
N_validation = len(InputData) - N_train

X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=N_validation)


###############################
#          モデル設定           #
###############################
# 重み行列作成
def weight_variable(shape, name=None):
    # scale:標準偏差
    # size:出力形状
    return np.random.normal(scale=.01, size=shape)


# 監視する値の変化が停止した時に訓練を終了する。
# monitorは監視する値(val_lossはmodel.fitの中に含まれる)
# patience:訓練が停止し、値が改善しなくなってからのエポック数
# verbose:冗長モード. 1ならon. onにしてあると標準出力に「ealry stopping」というコメントが表示される。
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

n_in = len(X[0][0])  # 入力の次元 = 1
n_hidden = 25
n_out = len(Y[0])  # 出力の次元 = 1

model = Sequential()
# model.add(SimpleRNN(units=n_hidden,
model.add(LSTM(units=n_hidden,
               kernel_initializer=weight_variable,  # kernel_initializer:重みの初期化方法
               input_dim=n_in,  # 入力の次元.入力層のノード数
               input_length=maxlen))  # 入力系列の長さ
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('linear'))  # y=αxのような線形関数

# http://qiita.com/shima_x/items/321e90564da38f3033b2
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['accuracy'])

###############################
#          モデル学習           #
###############################
his = model.fit(X_train, Y_train,
                batch_size=10,
                epochs=50,
                validation_data=(X_validation, Y_validation),
                callbacks=[early_stopping])

###############################
#            予測              #
###############################
truncate = maxlen
Z = X[:1]  # 元データの最初の一部だけ切り出し (1,50,1)

original = [f[i] for i in range(maxlen)]  # 1,50
predicted = [None for i in range(maxlen)]  # 1,50

for i in range(TrainDataLength - maxlen + 1):
    z_ = Z[-1:]  # 最後の1つ (1,50,1)
    y_ = model.predict(z_)  # 入力サンプルに対する予測値を出力 (1,1)
    sequence_ = np.concatenate((z_.reshape(maxlen, n_in)[1:], y_),  # z_.reshape(maxlen, n_in)[1:] (49,1)
                               axis=0).reshape(1, maxlen, n_in)  # (1,50,1)
    Z = np.append(Z, sequence_, axis=0)  # zを更新
    predicted.append(y_.reshape(-1))  # y_を１次元配列に変換

# print(np.array(predicted).shape) # 1,201

loss_and_metrics = model.evaluate(X_train, Y_train)
###############################
#        グラフで可視化         #
###############################
plt.figure()
plt.ylim([-1.5, 1.5])

plt.plot(NoiseCos(T, ampl=0, cycle=2), linestyle='dotted', color='#aaaaaa', label=u"正規のcos波")  # 正規のcos波
plt.plot(original, linestyle='dashed', color='black', label=u"ノイズ入りのcos派")  # ノイズ入りのcos派
plt.plot(predicted, color='red', label=u"学習データに基づくcos波")  # 学習データに基づくcos波

# fp = FontProperties(fname='/Users/cloudspider/Library/Fonts/NotoSansCJKjp-DemiLight.otf')
# plt.legend(prop=fp)

print('loss:{0}'.format(loss_and_metrics))
plt.show()
plot_model(model, to_file='model.png', show_shapes=True)
plot_history(his)
