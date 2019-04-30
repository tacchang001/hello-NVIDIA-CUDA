# https://s51517765.hatenadiary.jp/entry/2018/08/13/073000
# https://qiita.com/sasayabaku/items/b7872a3b8acc7d6261bf

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt


import matplotlib
matplotlib.use('TkAgg')


def sin(x, T):
    return np.sin(5 * np.pi * x / T) + np.sin(7 * np.pi * x / T) + np.sin(3 * np.pi * x / T) * 1.5
    # return np.sin(5 * np.pi * x / T)


def toy_problem(T, ampl=0.05):
    x = np.arange(0, 2 * T + 1)  # 等差数列のnumpy配列
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x, 100) + noise  # <class 'numpy.ndarray'>


def make_dataset(low_data, maxlen):  # maxlenを変更するとグラフがずれるので
    data, target = [], []
    for i in range(len(low_data) - maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])
        # print('------------')
        # print(data)
        # print('--')
        # print(target)
        # print('\n')

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target


if __name__ == "__main__":
    inputLength = 5
    TrainingTerm = 150
    f = toy_problem(T=TrainingTerm)
    # print('----')
    # print(f[:5])
    print(f[0:10])
    g, h = make_dataset(f, inputLength)
    print('----')
    print(g[:5])
    print('----')
    print(h[:5])
    future_test = g[170].T

    # 1つの学習データの時間の長さ
    time_length = future_test.shape[1]
    # 未来の予測データを保存していく変数
    future_result = np.empty((0))

    length_of_sequence = g.shape[1]
    in_out_neurons = 1  # 時刻にたいして出力が１つなので
    n_hidden = 500

    # # モデル構築
    # model = Sequential()
    # model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
    # model.add(Dense(in_out_neurons))
    # model.add(Activation("linear"))
    # optimizer = Adam(lr=0.001)  # 勾配手法
    # model.compile(loss="mean_squared_error", optimizer=optimizer)
    #
    # # 学習
    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    # model.fit(g, h, batch_size=200, epochs=300, validation_split=0.1, callbacks=[early_stopping])
    # # epochsはEarlyStopingによって短縮される
    # # batch_size 勾配の更新周期
    #
    # # 予測
    # predicted = model.predict(g)
    #
    # f2 = toy_problem(T=400)
    #
    # # 未来予想
    # for step2 in range(500):
    #     test_data = np.reshape(future_test, (1, time_length, 1))
    #     batch_predict = model.predict(test_data)
    #     future_test = np.delete(future_test, 0)
    #     future_test = np.append(future_test, batch_predict)
    #     future_result = np.append(future_result, batch_predict)
    #
    # # sin波をプロット
    # plt.figure()
    # plt.plot(range(inputLength, len(predicted) + inputLength), predicted, color="r",
    #          label="predict")  # maxlenのぶんオフセットしたところから
    # plt.plot(range(0, len(f2)), f2, color="k", label="row_data_2")  # range(start,end,関数名,)
    # plt.plot(range(0 + len(f), len(future_result) + len(f)), future_result, color="g", label="future")
    # plt.legend()
    # plt.show()
