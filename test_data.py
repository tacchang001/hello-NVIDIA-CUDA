import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')


def get_begin():
    return 0


def get_end(cycle):
    return 2 * np.pi * cycle


def synthetic_wave(x, T):
    return np.sin(5 * np.pi * x / T) + np.sin(7 * np.pi * x / T) + np.sin(3 * np.pi * x / T) * 1.5
    # return np.sin(5 * np.pi * x / T)


def synthetic_wave_data(cycle=1, n=100):
    _x = np.linspace(get_begin(), get_end(cycle), n)
    _y = pd.DataFrame({'wave': synthetic_wave(_x, n)})
    return _y


def sin_wave(begin=0, cycle=1, n=100):
    _x = np.linspace(begin, 2 * np.pi * cycle, n)
    _y = pd.Series(np.sin(_x), index=list(range(n)))
    _data = pd.DataFrame({'wave': _y})
    return _data


def noisy_sin_wave(cycle=1, n=100):
    # _s = sin_wave(begin, cycle, n)
    # _noise = pd.Series(_s['wave'] + 0.05 * np.random.randn(n), index=list(range(n)))
    _s = synthetic_wave_data(cycle, n)
    _noise = pd.Series(_s['wave'] + 0.05 * np.random.randn(n), index=list(range(n)))
    _n = pd.DataFrame({'noisy wave': _noise})
    _data = pd.concat([_s, _n], axis=1)
    return _data


def get_data_for_lstm():
    # windowを設定
    _WINDOW_LEN = 10

    # f = toy_problem(150)
    # plt.plot(f)
    # plt.show()

    df = noisy_sin_wave(cycle=120, n=200)
    df = (df - df.min()) / (df.max() - df.min())  # min-max normalization
    # print('----')
    # print(df)
    df.plot()
    plt.show()

    _lstm_in = []
    _data = pd.DataFrame({'noisy wave': df['noisy wave']})
    for i in range(len(_data) - _WINDOW_LEN):
        temp = _data[i:(i + _WINDOW_LEN)].copy()
        _lstm_in.append(temp)
    _lstm_out = df['wave'][_WINDOW_LEN:]

    _lstm_in = [np.array(_lstm_input) for _lstm_input in _lstm_in]
    _lstm_in = np.array(_lstm_in)
    _lstm_out = [np.array(_lstm_output) for _lstm_output in _lstm_out]
    _lstm_out = np.array(_lstm_out)

    return _lstm_in, _lstm_out


##############
# データ加工 #
##############

# 訓練データとテストデータへ切り分け
# n = df.shape[0]
# p = df.shape[1]
# train_start = 0
# train_end = int(np.floor(0.5 * n))
# test_start = train_end + 1
# test_end = n
# train = df.loc[np.arange(train_start, train_end), :]
# test = df.loc[np.arange(test_start, test_end), :]
#
# # LSTMへの入力用に処理（訓練）
# train_lstm_in, train_lstm_out = make_data_for_lstm(train)
#
# # LSTMへの入力用に処理（テスト）
# test_lstm_in, test_lstm_out = make_data_for_lstm(test)


if __name__ == "__main__":
    feature, target = get_data_for_lstm()
    print('actual:')
    print(type(feature))
    print(feature.shape)
    print(feature[0:2])
    print('expect:')
    print(type(target))
    print(target.shape)
    print(target[0:2])

    # print('---')
    # axis1 = feature[:, 0]
    # print(axis1)
    #
    # x = np.arange(0, len(axis1), 1)
    # plt.plot(x, axis1)
    #
    # print('---')
    # for i in np.arange(9):
    #     axis2 = feature[:, i]
    #     print(axis2)
    #     plt.plot(x, axis2)
    #
    # plt.show()
