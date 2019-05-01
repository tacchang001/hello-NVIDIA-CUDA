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


##############
# データ取得 #
##############

exp, obj = get_data_for_lstm()
# df.plot()
# plt.show()

########
# 予測 #
########

model = model_from_json(open('lstm.json', 'r').read())      # モデルの読み込み
model.load_weights('lstm.h5')                               # 重みの読み込み

predicted = model.predict(exp)

p = pd.DataFrame(predicted)
p.columns = ["predict"]
p['input'] = obj
print(p.head())
p.plot()
plt.show()
