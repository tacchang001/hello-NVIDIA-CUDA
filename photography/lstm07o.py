import numpy as np
from keras.models import model_from_json


def main():
    # モデルの読み込み
    model = model_from_json(open('and.json', 'r').read())

    # 重みの読み込み
    model.load_weights('and.h5')

    # 読み込んだ学習済みモデルで予測
    y = model.predict(np.array([[0, 1]]))
    print(y)  # [[ 0.17429274]]


if __name__ == '__main__':
    main()
