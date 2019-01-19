import os
os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(1, 4))
w1 = tf.Variable(tf.random_normal([4, 6], stddev=1))
w2 = tf.Variable(tf.random_normal([6, 2], stddev=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(x, feed_dict={x: [[0.8, 0.5, 0.2, 1.2]]}))
    print(sess.run(w1, feed_dict={x: [[0.8, 0.5, 0.2, 1.2]]}))
    print(sess.run(w2, feed_dict={x: [[0.8, 0.5, 0.2, 1.2]]}))
    print(sess.run(y, feed_dict={x: [[0.8, 0.5, 0.2, 1.2]]}))
