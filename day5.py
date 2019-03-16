import tensorflow as tf
import os

def csvread(filelist):
    """
    csv文件读取
    :param filelist: 文件路径+名字的列表
    :return:读取内容
    """
    # 文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 阅读器
    reader = tf.TextLineReader()
    # 读取文件
    key, value = reader.read(file_queue)
    # 指定读取内容的类型
    records = [['none'], ['none']]
    # 解码
    exemple, lable = tf.decode_csv(value,record_defaults=records)
    #批处理
    exemple_batch, lable_batch = tf.train.batch([exemple, lable],batch_size=9, num_threads=1, capacity=9)
    return exemple_batch, lable_batch


if __name__ == '__main__':
    filename = os.listdir('./temp/csvdata/')
    filelist = [os.path.join('./temp/csvdata/', file) for file in filename]
    exemple_batch, lable_batch = csvread(filelist)

    with tf.Session() as sess:
        # 线程管理器
        coord = tf.train.Coordinator()
        # 创建线程
        threads = tf.train.start_queue_runners(sess, coord=coord)
        print(sess.run([exemple_batch, lable_batch]))
        # 回收线程资源
        coord.request_stop()
        coord.join(threads)
