# https://qiita.com/hagi-suke/items/e4b0310ebfb73b12313e

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)  # MNIST読み込み時の警告を出力させないために書く

# 1.データセットの準備部分
mnist = input_data.read_data_sets("data/", one_hot=True)
train_images = mnist.train.images  # 訓練用の画像
train_labels = mnist.train.labels  # 訓練用の正解ラベル
test_images = mnist.test.images  # テスト用の画像
test_labels = mnist.test.labels  # テスト用の正解ラベル

# 2.データフローグラフ（設計図）の作成部分
learning_rate = 0.5  # 学習率
training_epochs = 1500  # エポック数
batch_size = 50  # ミニバッチのサイズ

# GPUを指定する
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0",  # GPU番号を指定
        allow_growth=True
    )
)

# 入力層
x = tf.placeholder(tf.float32, [None, 784])  # 入力データなど、実行時に確定する値はプレースホルダーで扱う
# 画像データの部分を28×28の行列に変換
img = tf.reshape(x, [-1, 28, 28, 1])
# 画像をログとして出力
tf.summary.image("input_data", img, 20)

# 隠れ層
with tf.name_scope("hidden"):
    # 重み（変数）
    w1 = tf.Variable(tf.truncated_normal([784, 64], stddev=0.1), name="w1")
    # バイアス（変数）
    b1 = tf.Variable(tf.zeros([64]), name="b1")
    # 活性化関数
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

# 出力層
with tf.name_scope("output"):
    # 重み（変数）
    w2 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), name="w2")
    # バイアス（変数）
    b2 = tf.Variable(tf.zeros([10]), name="b2")
    # 活性化関数
    out = tf.nn.softmax(tf.matmul(h1, w2) + b2)

# 　誤差関数
with tf.name_scope("loss"):
    # 正解ラベルも実行時に確定する値なのでプレースホルダーで扱う
    t = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.square(t - out))
    # 誤差をログとして出力
    tf.summary.scalar("loss", loss)

# 訓練（誤差逆伝播法アルゴリズムを選択）
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate  # 学習率
    ).minimize(loss)  # 最小化問題を解く

# 評価
with tf.name_scope("accuracy"):
    # (out＝t)の最大値を比較
    correct = tf.equal(tf.argmax(out, 1), tf.argmax(t, 1))
    # True(正解＝1)の平均を取る
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    # 精度をログとして出力
    tf.summary.scalar("accuracy", accuracy)

# 変数を初期化するノード
init = tf.global_variables_initializer()
# ログをマージするノード
summary_op = tf.summary.merge_all()

# 3.セッションで実行する部分
with tf.Session(config=config) as sess:
    # FileWriterオブジェクトを生成する
    summary_writer = tf.summary.FileWriter(
        "logs",  # イベントファイルの保存先
        sess.graph  # データフローグラフを視覚化する
    )
    sess.run(init)  # initノードを実行して変数を初期化
    for epoch in range(training_epochs):
        # ミニバッチを抽出
        train_images, train_labels = mnist.train.next_batch(batch_size)
        sess.run(
            train_step,  # 訓練を実行
            feed_dict={x: train_images,  # プレースホルダーxには訓練データのミニバッチをセット
                       t: train_labels}  # プレースホルダーtには訓練データの正解ラベルのミニバッチをセット
        )

        # 50回ごとにテストデータを使用して精度を出力
        epoch += 1
        if epoch % 50 == 0:
            acc_val = sess.run(
                accuracy,  # 評価を実行
                feed_dict={x: test_images,  # プレースホルダーxにはテストテータをセット
                           t: test_labels})  # プレースホルダーtにはテストデータの正解ラベルをセット
            print('(%d) accuracy = %.2f' % (epoch, acc_val))
            # イベントログをsummary_strに代入
            summary_str = sess.run(
                summary_op,  # ログをマージするノードを実行
                feed_dict={x: test_images,  # プレースホルダーxにはテストテータをセット
                           t: test_labels})  # プレースホルダーtにはテストデータの正解ラベルをセット
            # イベントファイルにログを追加
            summary_writer.add_summary(summary_str, epoch)

## $tensorboard --logdir="イベントファイルの入ったフォルダーのパス" でイベントログが確認出来るURLが与えられる
## 例：$tensorboard --logdir=./logs
## コマンド実行中にそのURL（例えばlocalhost:6006）にアクセスすれば良い
## Ctrl+Cで止められる
