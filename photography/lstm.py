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

##############
# データ作成 #
##############
np.random.seed(0)

_MAXLEN = 50


def NoiseCos(T=100, ampl=0.05, cycle=2):
    # amplifier:ノイズの程度
    # lowからhighまでのsize個の一様乱数を生成
    # cycle:描画するグラフの周期数
    t = np.arange(0, cycle * 2 * T + 1)
    return np.cos(2 * np.pi * t / T) + ampl * np.random.uniform(-1.0, 1.0, len(t))


def get_data():
    cycle = 2  # 描画する周期の個数
    T = 100  # 周期
    f = NoiseCos(T)  # f[t]でアクセス

    TrainDataLength = 2 * cycle * T  # 全時系列の長さ

    InputData = []  # 訓練データ
    TeacherData = []  # 教師データ

    for i in range(0, TrainDataLength - _MAXLEN + 1):
        InputData.append(f[i: _MAXLEN + i])
        TeacherData.append(f[i + _MAXLEN])

    X = np.array(InputData).reshape(len(InputData), _MAXLEN, 1)  # 訓練データ
    Y = np.array(TeacherData).reshape(len(TeacherData), 1)  # 教師データ
    # print(X.shape)  # 151,50,1
    # print(Y.shape)  # 151, 1

    # 訓練データ・検証データに分割(N_train:N_validation = 9:1)
    N_train = int(len(InputData) * 0.9)
    N_validation = len(InputData) - N_train

    return train_test_split(X, Y, test_size=N_validation)


# 重み行列作成
def weight_variable(shape, name=None):
    # scale:標準偏差
    # size:出力形状
    return np.random.normal(scale=.01, size=shape)


##############
# データ取得 #
##############

x_train, x_test, y_train, y_test = get_data()

##############
# モデル作成 #
##############

_TIME_STEPS = 5

_BATCH_SIZE = 10
_EPOCH_SIZE = 50
_HIDDEN = 25
_ACTIVATION = "tanh"

n_in = len(x_train[0][0])  # 入力の次元 = 1
n_out = len(y_train[0])  # 出力の次元 = 1

model = Sequential()
model.add(LSTM(units=_HIDDEN,
               kernel_initializer=weight_variable,  # kernel_initializer:重みの初期化方法
               input_dim=n_in,  # 入力の次元.入力層のノード数
               input_length=_MAXLEN))  # 入力系列の長さ
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('linear'))  # y=αxのような線形関数

##############
# 学習の実行 #
##############

# http://qiita.com/shima_x/items/321e90564da38f3033b2
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['accuracy'])

# 監視する値の変化が停止した時に訓練を終了する。
# monitorは監視する値(val_lossはmodel.fitの中に含まれる)
# patience:訓練が停止し、値が改善しなくなってからのエポック数
# verbose:冗長モード. 1ならon. onにしてあると標準出力に「ealry stopping」というコメントが表示される。
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

history = model.fit(x_train, y_train,
                    batch_size=_BATCH_SIZE,
                    epochs=_EPOCH_SIZE,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping])

model.save("lstm.model")
