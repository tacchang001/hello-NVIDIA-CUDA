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

from test_data import get_data_for_lstm


def draw_mae(model, history):
    """
    平均絶対誤差（MAE：Mean Absolute Error）グラフを描画する
    :param model:
    :param history:
    :return:
    """
    # MAEをプロットしてみよう
    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(history.epoch, history.history['loss'])
    ax1.set_title('TrainingError')

    if model.loss == 'mae':
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    else:
        ax1.set_ylabel('Model Loss', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    plt.show()


############
# 定数定義 #
############


##############
# データ取得 #
##############


exp, obj = get_data_for_lstm()

##############
# モデル作成 #
##############


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
    exp,
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
    exp, obj,
    epochs=50,
    batch_size=1,
    verbose=2,
    shuffle=True,
    callbacks=[early_stopping]
)

# 平均絶対誤差（MAE：Mean Absolute Error）
draw_mae(_model, _history)

open('lstm.json', "w").write(_model.to_json())      # モデルの保存
_model.save_weights('lstm.h5')                      # 学習済みの重みを保存

