# https://qiita.com/tizuo/items/b9af70e8cdc7fb69397f


import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('TkAgg')

# -------------------------------------------------------------

# データ読み込み　Yは最初の列に配置する
dataframe = pandas.read_csv('lstm_02_train.csv', usecols=[0, 3, 4, 5, 6], engine='python', skipfooter=1)
plt.plot(dataframe)
plt.show()

# -------------------------------------------------------------
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train = dataset
print(len(train))


# -------------------------------------------------------------

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


# convert an array of values into a dataset matrix
# if you give look_back 3, a part of the array will be like this: Jan, Feb, Mar
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        xset = []
        for j in range(dataset.shape[1]):
            a = dataset[i:(i + look_back), j]
            xset.append(a)
        dataY.append(dataset[i + look_back, 0])
        dataX.append(xset)
    return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 12
trainX, trainY = create_dataset(train, look_back)

# reshape input to be [samples, time steps(number of variables), features] *convert time series into column
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))

##########################################################################################
exit()
##########################################################################################

# -------------------------------------------------------------

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(trainX.shape[1], look_back)))  # shape：変数数、遡る時間数
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
_history = model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)

# -------------------------------------------------------------

# 平均絶対誤差（MAE：Mean Absolute Error）
draw_mae(model, _history)

open('lstm_02.json', "w").write(model.to_json())      # モデルの保存
model.save_weights('lstm_02.h5')                      # 学習済みの重みを保存
