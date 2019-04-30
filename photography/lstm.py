# LSTM
# http://www.algo-fx-blog.com/lstm-fx-predict/
# TensorflowでNN
# http://www.algo-fx-blog.com/tensorflow-neural-network-fx/

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib.font_manager import FontProperties
from keras.utils.vis_utils import plot_model

# from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM

from test_data import get_data

##############
# データ取得 #
##############

df = get_data()
print(df.shape)
print(df.columns)

##############
# データ加工 #
##############

n = df.shape[0]
p = df.shape[1]
# 訓練データとテストデータへ切り分け
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = df.loc[np.arange(train_start, train_end), :]
# print(data_train.tail())
data_test = df.loc[np.arange(test_start, test_end), :]
# print(data_test.tail())

# データの正規化
scalar = MinMaxScaler(feature_range=(-1, 1))
scalar.fit(data_train)
data_train_norm = scalar.transform(data_train)
# print(data_train_norm.shape)
# print(data_train_norm[0:2])
data_test_norm = scalar.transform(data_test)

# 特徴量とターゲットへ切り分け
X_train = data_train_norm[:, 1:]
# print(X_train[0:2])
y_train = data_train_norm[:, 0]
# print(y_train[0:2])
X_test = data_test_norm[:, 1:]
# print(X_test[0:2])
y_test = data_test_norm[:, 0]
# print(y_test[0:2])

# 正規化から通常の値へ戻す
y_test = y_test.reshape(19, 1)
test_inv = np.concatenate((y_test, X_test), axis=1)
test_inv = scalar.inverse_transform(test_inv)
# 正規化の前のテストデータ
# print(data_test.values[0])
# 正規化後のテストデータ
# print(y_test[0], X_test[0])
# 正規化から戻したデータ
# print(test_inv[0])

##############
# モデル作成 #
##############



# _TIME_STEPS = 5
#
# _BATCH_SIZE = 10
# _EPOCH_SIZE = 50
# _HIDDEN = 25
# _ACTIVATION = "tanh"
#
# n_in = len(x_train[0][0])  # 入力の次元 = 1
# n_out = len(y_train[0])  # 出力の次元 = 1
#
# model = Sequential()
# model.add(LSTM(units=_HIDDEN,
#                kernel_initializer=weight_variable,  # kernel_initializer:重みの初期化方法
#                input_dim=n_in,  # 入力の次元.入力層のノード数
#                input_length=_MAXLEN))  # 入力系列の長さ
# model.add(Dense(n_out, kernel_initializer=weight_variable))
# model.add(Activation('linear'))  # y=αxのような線形関数

##############
# 学習の実行 #
##############

# http://qiita.com/shima_x/items/321e90564da38f3033b2
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# model.compile(optimizer=optimizer,
#               loss='mean_squared_error',
#               metrics=['accuracy'])
#
# # 監視する値の変化が停止した時に訓練を終了する。
# # monitorは監視する値(val_lossはmodel.fitの中に含まれる)
# # patience:訓練が停止し、値が改善しなくなってからのエポック数
# # verbose:冗長モード. 1ならon. onにしてあると標準出力に「ealry stopping」というコメントが表示される。
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
#
# history = model.fit(x_train, y_train,
#                     batch_size=_BATCH_SIZE,
#                     epochs=_EPOCH_SIZE,
#                     validation_data=(x_test, y_test),
#                     callbacks=[early_stopping])
#
# model.save("lstm.model")
