# LSTM
# http://www.algo-fx-blog.com/lstm-fx-predict/
# TensorflowでNN
# http://www.algo-fx-blog.com/tensorflow-neural-network-fx/

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from test_data import get_data

############
# 定数定義 #
############

# windowを設定
_WINDOW_LEN = 10

##############
# データ取得 #
##############

df = get_data()
# print(df.shape)
# print(df.columns)
# print(df.head())

# del df['x']

##############
# データ加工 #
##############

# 訓練データとテストデータへ切り分け
n = df.shape[0]
p = df.shape[1]
train_start = 0
train_end = int(np.floor(0.8 * n))
test_start = train_end + 1
test_end = n
train = df.loc[np.arange(train_start, train_end), :]
# print(data_train.tail())
test = df.loc[np.arange(test_start, test_end), :]
# print(data_test.tail())

# train_lstm_in = []
# train_lstm_out = []
# for i in range(len(train) - _WINDOW_LEN):
#     temp = train[i:(i + _WINDOW_LEN)].copy()
#     # for col in train:
#     #     temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
#     train_lstm_in.append(temp)
# # lstm_train_out = (train['wave'][_WINDOW_LEN:].values / train['wave'][:-_WINDOW_LEN].values) - 1
# train_lstm_out = pd.DataFrame({'wave': train['wave'][_WINDOW_LEN:]})
#
# print(train.head(20))
# print('---')
# print(train_lstm_in[0])
# print('---')
# print(type(train_lstm_out))
# print(train_lstm_out.head(1))
# print('---')

# LSTMへの入力用に処理（訓練）
train_lstm_in = []
data_train = pd.DataFrame({'noisy wave': train['noisy wave']})
for i in range(len(data_train) - _WINDOW_LEN):
    temp = data_train[i:(i + _WINDOW_LEN)].copy()
    train_lstm_in.append(temp)
train_lstm_out = train['wave'][_WINDOW_LEN:]

# print('---')
# print(train_lstm_in[:5])
# print('---')
# print(train_lstm_out[:5])

# PandasのデータフレームからNumpy配列へ変換しましょう
train_lstm_in = [np.array(train_lstm_input) for train_lstm_input in train_lstm_in]
train_lstm_in = np.array(train_lstm_in)

# LSTMへの入力用に処理（テスト）
test_lstm_in = []
data_test = pd.DataFrame({'noisy wave': test['noisy wave']})
for i in range(len(data_test) - _WINDOW_LEN):
    temp = data_test[i:(i + _WINDOW_LEN)].copy()
    test_lstm_in.append(temp)
test_lstm_out = test['wave'][_WINDOW_LEN:]
test_lstm_in = [np.array(test_lstm_input) for test_lstm_input in test_lstm_in]
test_lstm_in = np.array(test_lstm_in)


##############
# モデル作成 #
##############


# def build_model(inputs, output_size, neurons, activ_func="linear",
#                 dropout=0.25, loss="mae", optimizer="adam"):
#     model = Sequential()
#     model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
#     model.add(Dropout(dropout))
#     model.add(Dense(units=output_size))
#     model.add(Activation(activ_func))
#     model.compile(loss=loss, optimizer=optimizer)
#     return model


def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


# ランダムシードの設定
np.random.seed(202)

# 初期モデルの構築
_model = build_model(
    train_lstm_in,
    output_size=1,
    neurons=20
)
early_stopping = EarlyStopping(
    monitor='loss',
    mode='min',
    patience=20
)
# データを流してフィッティングさせましょう
_history = _model.fit(
    train_lstm_in, train_lstm_out,
    epochs=50,
    batch_size=1,
    verbose=2,
    shuffle=True,
    callbacks=[early_stopping]
)

# MAEをプロットしてみよう
fig, ax1 = plt.subplots(1, 1)

ax1.plot(_history.epoch, _history.history['loss'])
ax1.set_title('TrainingError')

if _model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
else:
    ax1.set_ylabel('Model Loss', fontsize=12)
ax1.set_xlabel('# Epochs', fontsize=12)
plt.show()

_model.save("lstm.model")

########
# 予測 #
########

predicted = _model.predict(test_lstm_in)
in_d = pd.DataFrame(test_lstm_in)
in_d.plot()
plt.show()

print(predicted[:5])
dataf = pd.DataFrame(predicted)
dataf.columns = ["predict"]
dataf["input"] = test_lstm_out
dataf.plot()
plt.show()
