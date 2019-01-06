import os
import tensorflow as tf
import numpy as np
os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 8
SEED = 23455
STEPS = 8000
NUM_TESTS = 128
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_STEP = 5
LEARNING_RATE_DECAY = 0.99

rng = np.random.RandomState(SEED)
X = rng.rand(NUM_TESTS, 2)

Y = [[int(x0+x1 < 1)] for (x0, x1) in X]

print("initial input values: ")
print("[  VOLUME      WEIGHT  ]")
print(X)
print("initial output values: ")
print(Y)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 10], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([10, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

loss = tf.reduce_mean(tf.square(y-y_))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
train_step = tf.train.MomentumOptimizer(0.0005, 0.9).minimize(loss, global_step=global_step)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    print("initial w1 values: ")
    sess.run(init_op)
    print(sess.run(w1))
    print("initial w2 values: ")
    print(sess.run(w2))

    for i in range(STEPS):
        start = (i*BATCH_SIZE) % NUM_TESTS
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 200 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            learn = sess.run(learning_rate)
            print("After ", end="")
            print(i, end="")
            print(" training step(s), loss on all data is ", end="")
            print(total_loss, end="")
            print(", learning rate is ", end="")
            print(learn)

    print("final w1 values: ")
    print(sess.run(w1))
    print("final w2 values: ")
    print(sess.run(w2))
