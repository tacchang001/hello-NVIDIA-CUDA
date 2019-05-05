# https://qiita.com/tizuo/items/b9af70e8cdc7fb69397f

import numpy
import pandas
import matplotlib.pyplot as plt
import math
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('TkAgg')


# データ読み込み　Yは最初の列に配置する
dataframe = pandas.read_csv('lstm_02_test.csv', usecols=[0, 3, 4, 5, 6], engine='python', skipfooter=1)
# plt.plot(dataframe)
# plt.show()

# -------------------------------------------------------------
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
test = dataset
print(len(test))

# -------------------------------------------------------------


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


# -------------------------------------------------------------

# reshape into X=t and Y=t+1
look_back = 12
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps(number of variables), features] *convert time series into column
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))


model = model_from_json(open('lstm_02.json', 'r').read())      # モデルの読み込み
model.load_weights('lstm_02.h5')                               # 重みの読み込み

# -------------------------------------------------------------

# make predictions
testPredict = model.predict(testX)
pad_col = numpy.zeros(dataset.shape[1] - 1)


# invert predictions
def pad_array(val):
    return numpy.array([numpy.insert(pad_col, 0, x) for x in val])


testPredict = scaler.inverse_transform(pad_array(testPredict))
testY = scaler.inverse_transform(pad_array(testY))

# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % testScore)

# -------------------------------------------------------------

print(testY[:, 0])
print(testPredict[:, 0])
# shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[look_back + 1:len(testPredictPlot) + look_back, :] = testPredict
# plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(testPredictPlot)
plt.plot(testY[:, 0])
plt.plot(testPredict[:, 0])
plt.show()
