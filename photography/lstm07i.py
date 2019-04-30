# https://algorithm.joho.info/machine-learning/python-keras-save-load/

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop


def main():
    # 訓練データの用意
    # 入力データ
    x_train = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0]])
    # 入力の教師データ
    y_train = np.array([0.0, 0.0, 0.0, 1.0])

    # モデル構築
    model = Sequential()
    # 入力層
    model.add(Dense(2, activation='sigmoid', input_shape=(2,)))
    # 出力層
    model.add(Dense(1, activation='linear'))
    # コンパイル（勾配法：RMSprop、損失関数：mean_squared_error、評価関数：accuracy）
    model.compile(loss='mean_squared_error', optimizer=RMSprop(), metrics=['accuracy'])
    # 構築したモデルで学習
    history = model.fit(x_train, y_train, batch_size=4, epochs=3000)

    # モデルの性能評価
    score = model.evaluate(x_train, y_train, verbose=0)
    print('Score:', score[0])  # Score: 0.0209678225219
    print('Accuracy:', score[1])  # Accuracy: 1.0

    # モデルの保存
    open('and.json', "w").write(model.to_json())

    # 学習済みの重みを保存
    model.save_weights('and.h5')


if __name__ == '__main__':
    main()
