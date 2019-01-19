from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('./dataBase/', one_hot=True)
sess = tf.InteractiveSession()

INPUT = 784
HIDDEN = 300
OUTPUT = 10
w1 = tf.Variable(tf.truncated_normal([INPUT, HIDDEN], stddev = 0.1))
b1 = tf.Variable(tf.zeros([HIDDEN]))
w2 = tf.Variable(tf.truncated_normal([HIDDEN, OUTPUT], stddev = 0.1))
b2 = tf.Variable(tf.zeros([OUTPUT]))

x = tf.placeholder(tf.float32, [None, INPUT])
keep_prob = tf.placeholder(tf.float32)          # a probability of dropping nodes

hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    losses, step = sess.run([loss, train_step], feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 0.75})
    acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0})
    print("The current training step is: ", end="")
    print(step, end="")
    print("; losses is: ", end="")
    print(losses, end="")
    print("; accuracy is: ", end="")
    print(acc)
