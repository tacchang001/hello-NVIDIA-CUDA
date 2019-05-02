# LSTM
# http://www.algo-fx-blog.com/lstm-fx-predict/
# TensorflowでNN
# http://www.algo-fx-blog.com/tensorflow-neural-network-fx/

import pandas as pd
from keras.models import model_from_json
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

model = model_from_json(open('lstm_01.json', 'r').read())      # モデルの読み込み
model.load_weights('lstm_01.h5')                               # 重みの読み込み

predicted = model.predict(exp)

p = pd.DataFrame(predicted)
p.columns = ["predict"]
p['input'] = obj
print(p.head())
p.plot()
plt.show()
