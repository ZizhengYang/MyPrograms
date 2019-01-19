import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'

TOTAL_SIZE = 300
BATCH_SIZE = 30
SEED = 2
STEPS = 40000

rng = np.random.RandomState(SEED)
X = rng.randn(TOTAL_SIZE, 2)
Y_ = [int(a * a + b * b < 2) for (a, b) in X]
Y_color = [['red' if y else 'blue'] for y in Y_]

print("X values(a, b):")
print(X)
print("Y_ values(whether (a,b) is red or blue):")
print(Y_)

X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)

print("X values(a, b):")
print(X)
print("Y_ values(whether (a,b) is red or blue):")
print(Y_)

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_color))
plt.show()

print("Y_color values(whether (a,b) is red or blue):")
print(Y_color)


def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2

losses = tf.reduce_mean(tf.square(y-y_))
regularized_loss = losses + tf.add_n(tf.get_collection('losses'))

train_step_a = tf.train.AdamOptimizer(0.0001).minimize(losses)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(STEPS):
        start = (i * BATCH_SIZE) % TOTAL_SIZE
        end = start + BATCH_SIZE
        sess.run(train_step_a, feed_dict={x: X[start: end], y_: Y_[start: end]})

        if i % 2000 == 0:
            total_loss_a = sess.run(losses, feed_dict={x: X, y_: Y_})
            print("After ", end="")
            print(i, end="")
            print(" training step(s), loss is ", end="")
            print(total_loss_a)

    xx, yy = np.mgrid[-3: 3: 0.01, -3: 3: 0.01]

    '''
    [[-3.   -3.   -3.   ... -3.   -3.   -3.  ]
     [-2.99 -2.99 -2.99 ... -2.99 -2.99 -2.99]
     [-2.98 -2.98 -2.98 ... -2.98 -2.98 -2.98]
                        ...
     [ 2.97  2.97  2.97 ...  2.97  2.97  2.97]
     [ 2.98  2.98  2.98 ...  2.98  2.98  2.98]
     [ 2.99  2.99  2.99 ...  2.99  2.99  2.99]]

    [[-3.   -2.99 -2.98 ...  2.97  2.98  2.99]
     [-3.   -2.99 -2.98 ...  2.97  2.98  2.99]
     [-3.   -2.99 -2.98 ...  2.97  2.98  2.99]
                        ...
     [-3.   -2.99 -2.98 ...  2.97  2.98  2.99]
     [-3.   -2.99 -2.98 ...  2.97  2.98  2.99]
     [-3.   -2.99 -2.98 ...  2.97  2.98  2.99]]
    '''

    grid_a = np.c_[xx.ravel(), yy.ravel()]
    probs_a = sess.run(y, feed_dict={x:grid_a})
    probs_a = probs_a.reshape(xx.shape)

    print("w1 is shown here:")
    print(sess.run(w1))
    print("b1 is shown here:")
    print(sess.run(b1))
    print("w2 is shown here:")
    print(sess.run(w2))
    print("b2 is shown here:")
    print(sess.run(b2))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_color))
plt.contour(xx, yy, probs_a, levels=[0.5])
plt.show()


train_step_b = tf.train.AdamOptimizer(0.0001).minimize(regularized_loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(STEPS):
        start = (i * BATCH_SIZE) % TOTAL_SIZE
        end = start + BATCH_SIZE
        sess.run(train_step_b, feed_dict={x: X[start: end], y_: Y_[start: end]})

        if i % 2000 == 0:
            total_loss_b = sess.run(regularized_loss, feed_dict={x: X, y_: Y_})
            print("After ", end="")
            print(i, end="")
            print(" training step(s), loss is ", end="")
            print(total_loss_b)

    xx, yy = np.mgrid[-3: 3: 0.01, -3: 3: 0.01]
    grid_b = np.c_[xx.ravel(), yy.ravel()]
    probs_b = sess.run(y, feed_dict={x: grid_b})
    probs_b = probs_b.reshape(xx.shape)

    print("w1 is shown here:")
    print(sess.run(w1))
    print("b1 is shown here:")
    print(sess.run(b1))
    print("w2 is shown here:")
    print(sess.run(w2))
    print("b2 is shown here:")
    print(sess.run(b2))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_color))
plt.contour(xx, yy, probs_b, levels=[0.5])
plt.show()
