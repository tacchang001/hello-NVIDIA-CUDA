# https://qiita.com/hagi-suke/items/e4b0310ebfb73b12313e

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions

# 1.データセットを用意する
train, test = chainer.datasets.get_mnist(ndim=3)  # ndim=width,height,color

batch_size = 10  # バッチサイズ
uses_device = 0  # GPU#0を使用(CPU=-1)


# 2.モデルの作成
class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 64)
            self.l2 = L.Linear(None, 10)

    def __call__(self, x, t=None, train=True):
        h = F.relu(self.l1(x))
        h = self.l2(h)
        # 訓練時には損失を、テスト時には結果を返す
        return F.softmax_cross_entropy(h, t) if train else F.softmax(h)


# 3.Updater（パラメータ更新手法）の作成
# ここではStandardUpdaterを使用するため、必要ない

# 4.実行する
# モデルを作成
model = MLP()

# GPUに対応させる
if uses_device >= 0:
    chainer.cuda.get_device_from_id(uses_device).use()
    chainer.cuda.check_cuda_available()
    # GPU用データ形式に変換
    model.to_gpu()

# iterator（繰り返し条件）を作成する
train_iter = iterators.SerialIterator(train, batch_size, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

# 誤差逆伝播法アルゴリズムを選択する
optimizer = optimizers.Adam()
optimizer.setup(model)

# デバイスを選択してTrainer（訓練）を作成する
updater = training.StandardUpdater(train_iter, optimizer, device=uses_device)
trainer = training.Trainer(updater, (100, 'epoch'), out="result")
# Trainerにはextensionという便利なオプションがつけられる
trainer.extend(extensions.Evaluator(test_iter, model, device=uses_device))  # テストをTrainerに設定する
trainer.extend(extensions.ProgressBar())  # 学習の進展を表示するようにする

# 機械学習を実行する
trainer.run()
