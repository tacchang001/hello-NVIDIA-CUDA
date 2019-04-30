import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

_MAX_LEN = 50
_CYCLE = 2  # 描画する周期の個数
_PERIOD = 100  # 周期


def noisy_cos(T=_PERIOD, ampl=0.05, cycle=2):
    """
    ノイズ入りcos波
    :param T:
    :param ampl: ノイズの程度
    :param cycle: 描画するグラフの周期数
    :return:
    """
    t = np.arange(0, cycle * 2 * T + 1)
    return np.cos(2 * np.pi * t / T) + ampl * np.random.uniform(-1.0, 1.0, len(t))


def sin_cos(T=_PERIOD):
    x = np.arange(0, 2 * T + 1)
    # print(x)
    # plt.plot(x, linestyle='dotted', color='#aaaaaa', label=u"正規のcos波")
    # plt.show()
    # print(x.shape)  # (201,)

    _data = np.concatenate(
        [
            np.sin(2.0 * np.pi * x / T).reshape(-1, 1),
            np.cos(2.0 * np.pi * x / T).reshape(-1, 1)
        ],
        axis=1)
    # print(_data)
    # plt.plot(_data, linestyle='dotted', color='#aaaaaa', label=u"正規のcos波")
    # plt.show()
    # print(_data.shape)  # (201, 2)
    return _data


def get_data_dont_use():
    # データセット作り
    _data = sin_cos()

    timesteps = 10
    data_dim = _data.shape[1]

    lstm_data = []
    index_data = []

    for i in range(timesteps):
        length = _data[i:-1].shape[0] // timesteps
        lstm_data.append(_data[i:i + length * timesteps].reshape(length, timesteps, data_dim))
        index_data.append(np.arange(i, i + (length * timesteps), timesteps))

    lstm_data = np.concatenate(lstm_data, axis=0)
    print(type(lstm_data))
    plt.plot(lstm_data[:, 0], linestyle='dotted', color='#aaaaaa', label=u"正規のcos波")  # 正規のcos波
    plt.show()

    index_data = np.concatenate(index_data, axis=0)
    lstm_data = lstm_data[pd.Series(index_data).sort_values().index]
    # plt.plot(lstm_data[0][0], linestyle='dotted', color='#aaaaaa', label=u"正規のcos波")  # 正規のcos波
    # plt.show()

    lstm_data_x = lstm_data[:, :-1, :]
    lstm_data_y = lstm_data[:, -1, :]

    return lstm_data


def get_data():
    return noisy_sin_wave(begin=0, cycle=4, n=100)


def sin_wave(begin=0, cycle=1, n=100):
    _x = np.linspace(begin, 2 * np.pi * cycle, n)
    _y = pd.Series(np.sin(_x), index=list(range(n)))
    _data = pd.DataFrame({'x': _x, 'wave': _y})
    return _data


def noisy_sin_wave(begin=0, cycle=1, n=100):
    _s = sin_wave(begin, cycle, n)
    _noise = pd.Series(np.sin(_s['wave']) + 0.1 * np.random.randn(n), index=list(range(n)))
    _n = pd.DataFrame({'noisy wave': _noise})
    _data = pd.concat([_s, _n], axis=1)
    return _data


if __name__ == "__main__":
    x = get_data()
    print(x.shape)
    print(x.head())
    # x.plot(color=('r', 'b', 'g'))
    # plt.show()
