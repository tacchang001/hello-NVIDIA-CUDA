# https://qiita.com/hikobotch/items/018808ef795061176824

# とりあえず片っ端からimport
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import time
from matplotlib import pyplot as plt

uses_device = 0  # GPU#0を使用(CPU=-1)


# データ
def get_dataset(N):
    x = np.linspace(0, 2 * np.pi, N)
    y = np.sin(x)
    return x, y


# ニューラルネットワーク
class MyChain(Chain):
    def __init__(self, n_units=10):
        super(MyChain, self).__init__(
            l1=L.Linear(1, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, 1))

    def __call__(self, x_data, y_data):
        x = Variable(x_data.astype(np.float32).reshape(len(x_data), 1))  # Variableオブジェクトに変換
        y = Variable(y_data.astype(np.float32).reshape(len(y_data), 1))  # Variableオブジェクトに変換
        return F.mean_squared_error(self.predict(x), y)

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        return h3

    def get_predata(self, x):
        return self.predict(Variable(x.astype(np.float32).reshape(len(x), 1))).data


# main
if __name__ == "__main__":

    # 学習データ
    N = 1000
    x_train, y_train = get_dataset(N)

    # テストデータ
    N_test = 900
    x_test, y_test = get_dataset(N_test)

    # 学習パラメータ
    batchsize = 10
    n_epoch = 500
    n_units = 100

    # モデル作成
    model = MyChain(n_units)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # GPUに対応させる
    if uses_device >= 0:
        chainer.cuda.get_device_from_id(uses_device).use()
        chainer.cuda.check_cuda_available()
        # GPU用データ形式に変換
        model.to_gpu()

    # 学習ループ
    train_losses = []
    test_losses = []
    print("start...")
    start_time = time.time()
    for epoch in range(1, n_epoch + 1):

        # training
        perm = np.random.permutation(N)
        sum_loss = 0
        for i in range(0, N, batchsize):
            x_batch = x_train[perm[i:i + batchsize]]
            y_batch = y_train[perm[i:i + batchsize]]

            model.zerograds()
            loss = model(x_batch, y_batch)
            sum_loss += loss.data * batchsize
            loss.backward()
            optimizer.update()

        average_loss = sum_loss / N
        train_losses.append(average_loss)

        # test
        loss = model(x_test, y_test)
        test_losses.append(loss.data)

        # 学習過程を出力
        if epoch % 10 == 0:
            print("epoch: {}/{} train loss: {} test loss: {}".format(epoch, n_epoch, average_loss, loss.data))

        # 学習結果のグラフ作成
        if epoch in [10, 500]:
            theta = np.linspace(0, 2 * np.pi, N_test)
            sin = np.sin(theta)
            test = model.get_predata(theta)
            plt.plot(theta, sin, label="sin")
            plt.plot(theta, test, label="test")
            plt.legend()
            plt.grid(True)
            plt.xlim(0, 2 * np.pi)
            plt.ylim(-1.2, 1.2)
            plt.title("sin")
            plt.xlabel("theta")
            plt.ylabel("amp")
            plt.savefig("fig/fig_sin_epoch{}.png".format(epoch))  # figフォルダが存在していることを前提
            plt.clf()

    print("end")

    interval = int(time.time() - start_time)
    print("実行時間: {}sec".format(interval))

    # 誤差のグラフ作成
    plt.plot(train_losses, label="train_loss")
    plt.plot(test_losses, label="test_loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("fig/fig_loss.png")  # figフォルダが存在していることを前提
    plt.clf()

# https://qiita.com/chachay/items/052406176c55dd5b9a6a

# import numpy as np
#
# import chainer
# import chainer.functions as F
# import chainer.links as L
# from chainer import report, training, Chain, datasets, iterators, optimizers
# from chainer.training import extensions
# from chainer.datasets import tuple_dataset
#
# import matplotlib.pyplot as plt
#
#
# class MLP(Chain):
#     n_input = 1
#     n_output = 1
#     n_units = 5
#
#     def __init__(self):
#         super(MLP, self).__init__(
#             l1=L.Linear(self.n_input, self.n_units),
#             l2=L.LSTM(self.n_units, self.n_units),
#             l3=L.Linear(self.n_units, self.n_output),
#         )
#
#     def reset_state(self):
#         self.l2.reset_state()
#
#     def __call__(self, x):
#         h1 = self.l1(x)
#         h2 = self.l2(h1)
#         return self.l3(h2)
#
#
# class LossFuncL(Chain):
#     def __init__(self, predictor):
#         super(LossFuncL, self).__init__(predictor=predictor)
#
#     def __call__(self, x, t):
#         x.data = x.data.reshape((-1, 1)).astype(np.float32)
#         t.data = t.data.reshape((-1, 1)).astype(np.float32)
#
#         y = self.predictor(x)
#         loss = F.mean_squared_error(y, t)
#         report({'loss': loss}, self)
#         return loss
#
#
# class LSTM_test_Iterator(chainer.dataset.Iterator):
#     def __init__(self, dataset, batch_size=10, seq_len=5, repeat=True):
#         self.seq_length = seq_len
#         self.dataset = dataset
#         self.nsamples = len(dataset)
#
#         self.batch_size = batch_size
#         self.repeat = repeat
#
#         self.epoch = 0
#         self.iteration = 0
#         self.offsets = np.random.randint(0, len(dataset), size=batch_size)
#
#         self.is_new_epoch = False
#
#     def __next__(self):
#         if not self.repeat and self.iteration * self.batch_size >= self.nsamples:
#             raise StopIteration
#
#         x, t = self.get_data()
#         self.iteration += 1
#
#         epoch = self.iteration // self.batch_size
#         self.is_new_epoch = self.epoch < epoch
#         if self.is_new_epoch:
#             self.epoch = epoch
#             self.offsets = np.random.randint(0, self.nsamples, size=self.batch_size)
#
#         return list(zip(x, t))
#
#     @property
#     def epoch_detail(self):
#         return self.iteration * self.batch_size / len(self.dataset)
#
#     def get_data(self):
#         tmp0 = [self.dataset[(offset + self.iteration) % self.nsamples][0]
#                 for offset in self.offsets]
#         tmp1 = [self.dataset[(offset + self.iteration + 1) % self.nsamples][0]
#                 for offset in self.offsets]
#         return tmp0, tmp1
#
#     def serialzie(self, serialzier):
#         self.iteration = serializer('iteration', self.iteration)
#         self.epoch = serializer('epoch', self.epoch)
#
#
# class LSTM_updater(training.StandardUpdater):
#     def __init__(self, train_iter, optimizer, device):
#         super(LSTM_updater, self).__init__(train_iter, optimizer, device=device)
#         self.seq_length = train_iter.seq_length
#
#     def update_core(self):
#         loss = 0
#
#         train_iter = self.get_iterator('main')
#         optimizer = self.get_optimizer('main')
#
#         for i in range(self.seq_length):
#             batch = np.array(train_iter.__next__()).astype(np.float32)
#             x, t = batch[:, 0].reshape((-1, 1)), batch[:, 1].reshape((-1, 1))
#             loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))
#
#         optimizer.target.zerograds()
#         loss.backward()
#         loss.unchain_backward()
#         optimizer.update()
#
#
# model = LossFuncL(MLP())
# optimizer = optimizers.Adam()
# optimizer.setup(model)
#
# # データ作成
# N_data = 100
# N_Loop = 3
# t = np.linspace(0., 2 * np.pi * N_Loop, num=N_data)
#
# X = 0.8 * np.sin(2.0 * t)
#
# # データセット
# N_train = int(N_data * 0.8)
# N_test = int(N_data * 0.2)
#
# tmp_DataSet_X = np.array(X).astype(np.float32)
#
# x_train, x_test = np.array(tmp_DataSet_X[:N_train]), np.array(tmp_DataSet_X[N_train:])
#
# train = tuple_dataset.TupleDataset(x_train)
# test = tuple_dataset.TupleDataset(x_test)
#
# train_iter = LSTM_test_Iterator(train, batch_size=10, seq_len=10)
# test_iter = LSTM_test_Iterator(test, batch_size=10, seq_len=10, repeat=False)
#
# updater = LSTM_updater(train_iter, optimizer, -1)
# trainer = training.Trainer(updater, (1000, 'epoch'), out='result')
#
# eval_model = model.copy()
# eval_rnn = eval_model.predictor
# eval_rnn.train = False
# trainer.extend(extensions.Evaluator(
#     test_iter, eval_model, device=-1,
#     eval_hook=lambda _: eval_rnn.reset_state()))
#
# trainer.extend(extensions.LogReport())
#
# trainer.extend(
#     extensions.PrintReport(
#         ['epoch', 'main/loss', 'validation/main/loss']
#     )
# )
#
# trainer.extend(extensions.ProgressBar())
#
# trainer.run()
