import tensorflow as tf
import os

# a = tf.constant(6.0)
# b = tf.constant(5.0)
#
# sum = tf.add(a,b)
#
# print(sum)
# graph = tf.get_default_graph()
# print(graph)
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     print(sess.run(sum))


# g = tf.Graph
# # print(g)
# #
# # with g.as_default():
# #     c = tf.constant(11.0)
# #     print(c.graph)

tf.app.flags.DEFINE_integer('max_step', 100, '模型训练最大步数')
tf.app.flags.DEFINE_string('model_dir', '', '模型存储路径')
FLAGS = tf.app.flags.FLAGS

def myregression():
    """
    自定义线性回归
    :return: 0
    """
    with tf.variable_scope('data'):
        # 准备数据
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
        y = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope('model'):
        # 构造模型，一个特征值，一个权重，一个偏置
        weight = tf.Variable(tf.random_normal([1, 1], mean=0, stddev=1.0), name='w')
        bias = tf.Variable(0.0, name='b')

        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope('loss'):
        # 损失函数，均方误差
        loss = tf.reduce_mean(tf.square(y - y_predict))

    with tf.variable_scope('optimizer'):
        # 梯度下降优化损失
        tran_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    tf.summary.scalar('losser', loss)
    tf.summary.histogram('weighter', weight)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    #初始化变量op
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op )
        print(f'初始化的权重:{weight.eval()},偏置:{bias.eval()}')
        filewriter = tf.summary.FileWriter('./temp/test/', graph=sess.graph)

        if os.path.exists('./temp/ckpt/checkpoint'):
            saver.restore(sess, FLAGS.model_dir)

        for i in range(FLAGS.max_step):
            summary = sess.run(merged)
            sess.run(tran_op)
            filewriter.add_summary(summary, i)
            print(f'优化的权重：{weight.eval()},偏置:{bias.eval()}')

        saver.save(sess, FLAGS.model_dir)

    return 0


if __name__ == '__main__':

    myregression()
