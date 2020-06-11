import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.disable_eager_execution()

# 实现一个线性回归预测
with tf.variable_scope("data"):
    # 1、准备数据，x 特征值 [100, 1]   y 目标值[100]
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
    # 矩阵相乘必须是二维的
    y_true = tf.matmul(x, [[0.7]]) + 0.8

# 2、建立线性回归模型 1个特征，1个权重， 一个偏置 y = x w + b
with tf.variable_scope("model"):
    weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name='weight')
    bias = tf.Variable(0.0, name="bias")
    y_predict = tf.matmul(x, weight) + bias

# 3、建立损失函数，均方误差
with tf.variable_scope("loss"):
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

# 4、梯度下降优化损失 leaning_rate: 0 ~ 1, 2, 3,5, 7, 10
with tf.variable_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(loss)

# 1、收集tensor
tf.summary.scalar("losses", loss)
tf.summary.histogram("weights", weight)

# 定义合并tensor的op
merged = tf.summary.merge_all()

# 定义一个初始化变量的op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # 建立事件文件
    file_writer = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

    # 循环训练 运行优化
    for i in range(5000):
        sess.run(train_op)
        # 运行合并的tensor
        summary = sess.run(merged)
        file_writer.add_summary(summary, i)
        print("第%d次优化后，weight=%f，bias=%f" % (i, weight.eval(), bias.eval()))

# tensorboard --logdir=./tmp/summary/test/
# 查看结果


