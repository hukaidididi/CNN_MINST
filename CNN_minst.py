import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

minst = input_data.read_data_sets('MINST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 计算正确率属于test，不需要dropout，所以设keep_prob=1
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)   # 保留率，只在train时用；测试时全部用，即test时设keep_prob=1
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # -1相当于None，先不管有多少组数据；因为minst是黑白的，所以channel=1

# # cov1
W_cov1 = weight_variable([5, 5, 1, 32])
b_cov1 = biases_variable([32])
hidden_cov1 = tf.nn.relu(conv2d(x_image, W_cov1) + b_cov1)  # 28*28*32
hidden_pool1 = max_pool(hidden_cov1)  # 14*14*32

# # cov2
W_cov2 = weight_variable([5, 5, 32, 64])
b_cov2 = biases_variable([64])
hidden_cov2 = tf.nn.relu(conv2d(hidden_pool1, W_cov2) + b_cov2)  # 14*14*64
hidden_pool2 = max_pool(hidden_cov2)  # 7*7*64

# # fc1
hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7*7*64])
# 将第二层卷积层输出的三维向量（shape为[7,7,64]）flatten压平成一维向量，size为7*7*64
W_fc1 = weight_variable([7*7*64, 1024])  # fc1层的输入为7*7*64，设定输出为1024
b_fc1 = biases_variable([1024])
hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, W_fc1) + b_fc1)
hidden_fc1_drop = tf.nn.dropout(hidden_fc1, keep_prob)

# # fc2
W_fc2 = weight_variable([1024, 10])
b_fc2 = biases_variable([10])
prediction = tf.nn.softmax(tf.matmul(hidden_fc1_drop, W_fc2) + b_fc2)

Cross_Entropy = tf.reduce_mean(tf.reduce_sum(-ys*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(Cross_Entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

start_time = time.time()
for i in range(1000):
    batch_xs, batch_ys = minst.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 100 == 0:
        print('epoch%d: ' % int((i / 100)+1), (compute_accuracy(v_xs=minst.test.images[:1000], v_ys=minst.test.labels[:1000])))

duration = time.time() - start_time
print('总用时为：',  duration, '秒')
