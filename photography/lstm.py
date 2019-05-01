# LSTM
# http://www.algo-fx-blog.com/lstm-fx-predict/
# TensorflowでNN
# http://www.algo-fx-blog.com/tensorflow-neural-network-fx/

import numpy as np
import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from test_data import get_data


def draw_mae(history):
    """
    平均絶対誤差（MAE：Mean Absolute Error）グラフを描画する
    :param history:
    :return:
    """
    # MAEをプロットしてみよう
    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(history.epoch, history.history['loss'])
    ax1.set_title('TrainingError')

    if _model.loss == 'mae':
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    else:
        ax1.set_ylabel('Model Loss', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    plt.show()

############
# 定数定義 #
############

# windowを設定
_WINDOW_LEN = 10

##############
# データ取得 #
##############

df = get_data()
df.plot()
plt.show()

##############
# データ加工 #
##############


def make_data_for_lstm(in_data):
    _lstm_in = []
    _data = pd.DataFrame({'noisy wave': in_data['noisy wave']})
    for i in range(len(_data) - _WINDOW_LEN):
        temp = _data[i:(i + _WINDOW_LEN)].copy()
        _lstm_in.append(temp)
    _lstm_in = [np.array(_lstm_input) for _lstm_input in _lstm_in]
    _lstm_in = np.array(_lstm_in)
    _lstm_out = in_data['wave'][_WINDOW_LEN:]

    return _lstm_in, _lstm_out


# 訓練データとテストデータへ切り分け
n = df.shape[0]
p = df.shape[1]
train_start = 0
train_end = int(np.floor(0.5 * n))
test_start = train_end + 1
test_end = n
train = df.loc[np.arange(train_start, train_end), :]
test = df.loc[np.arange(test_start, test_end), :]

# LSTMへの入力用に処理（訓練）
train_lstm_in, train_lstm_out = make_data_for_lstm(train)

# LSTMへの入力用に処理（テスト）
test_lstm_in, test_lstm_out = make_data_for_lstm(test)

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

# 平均絶対誤差（MAE：Mean Absolute Error）
# draw_mae(_history)

open('lstm.json', "w").write(_model.to_json())      # モデルの保存
_model.save_weights('lstm.h5')                      # 学習済みの重みを保存

########
# 予測 #
########

model = model_from_json(open('lstm.json', 'r').read())       # モデルの読み込み
model.load_weights('lstm.h5')                               # 重みの読み込み

predicted = model.predict(test_lstm_in)

dataf = pd.DataFrame(predicted)
dataf.columns = ["predict"]
dataf["input"] = test_lstm_out
dataf.plot()
plt.show()
