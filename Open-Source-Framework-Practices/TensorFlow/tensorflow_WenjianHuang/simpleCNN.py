from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('./mnist_dataBase/', one_hot=True)
MODEL_SAVE_PATH = "./cnn_model/"
MODEL_NAME = "mnist_model_cnn"
BATCH_SIZE = 50
TOTAL_BATCHES = mnist.train.num_examples // BATCH_SIZE
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
Train_Rounds = 20000
sess = tf.InteractiveSession()


def weiht_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(shape=shape, value=0.1))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


global_step = tf.Variable(0, trainable=False)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

w_conv1 = weiht_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weiht_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weiht_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weiht_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    TOTAL_BATCHES,
    LEARNING_RATE_DECAY,
    staircase=True
)

# ce = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])
# cem = tf.reduce_mean(ce)
# loss = cem + tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
ema_op = ema.apply(tf.trainable_variables())
with tf.control_dependencies([train_step, ema_op]):
    train_op = tf.no_op(name="train")

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

Loss = []
Accuracy = []

for i in range(Train_Rounds):
    batch = mnist.train.next_batch(BATCH_SIZE)
    train, step_value = sess.run([train_op, global_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 100 == 0:
        train_accuracy, loss_value = sess.run([accuracy, loss], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("Step: ", end="")
        print(i, end="")
        print(" <==> Loss value is: ", end="")
        print(loss_value, end="")
        print(" <==> Training accuracy: ", end="")
        print(train_accuracy)
    Loss.append(loss_value)
    Accuracy.append(train_accuracy)

print("----------------Final Total Accuracy: ", end="")
total_acc = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print(total_acc, end="")
print("----------------")
saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cross Loss')
plt.plot(Loss)
plt.grid()
plt.show()

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy')
plt.plot(Accuracy)
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(2, 2))
ax.imshow(np.reshape(mnist.test.images[0], (28, 28)))
plt.show()

input_image = mnist.test.images[0:1]

conv1_32 = sess.run(h_conv1, feed_dict={x:input_image})
conv1_transpose = sess.run(tf.transpose(conv1_32, [3, 0, 1, 2]))
fig3, ax3 = plt.subplots(nrows=4, ncols=8, figsize=(8, 4))
for i in range(8):
    ax3[0][i].imshow(conv1_transpose[i][0])
    ax3[1][i].imshow(conv1_transpose[i+8][0])
    ax3[2][i].imshow(conv1_transpose[i+16][0])
    ax3[3][i].imshow(conv1_transpose[i+24][0])
plt.title('Conv1 32x28x28')
plt.show()

pool1_32 = sess.run(h_pool1, feed_dict={x:input_image})
pool1_transpose = sess.run(tf.transpose(pool1_32, [3, 0, 1, 2]))
fig4, ax4 = plt.subplots(nrows=4, ncols=8, figsize=(8, 4))
for i in range(8):
    ax4[0][i].imshow(pool1_transpose[i][0])
    ax4[1][i].imshow(pool1_transpose[i + 8][0])
    ax4[2][i].imshow(pool1_transpose[i + 16][0])
    ax4[3][i].imshow(pool1_transpose[i + 24][0])
plt.title('Pool1 32x14x14')
plt.show()

conv2_64 = sess.run(h_conv2, feed_dict={x:input_image})
conv2_transpose = sess.run(tf.transpose(conv2_64, [3, 0, 1, 2]))
fig5, ax5 = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
for i in range(8):
    ax5[0][i].imshow(conv2_transpose[i][0])
    ax5[1][i].imshow(conv2_transpose[i + 8][0])
    ax5[2][i].imshow(conv2_transpose[i + 16][0])
    ax5[3][i].imshow(conv2_transpose[i + 24][0])
    ax5[4][i].imshow(conv2_transpose[i + 32][0])
    ax5[5][i].imshow(conv2_transpose[i + 40][0])
    ax5[6][i].imshow(conv2_transpose[i + 48][0])
    ax5[7][i].imshow(conv2_transpose[i + 56][0])
plt.title('Conv2 64x14x14')
plt.show()

pool2_64 = sess.run(h_pool2, feed_dict={x:input_image})         #[1, 7, 7, 32]
pool2_transpose = sess.run(tf.transpose(pool2_64, [3, 0, 1, 2]))
fig6, ax6 = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
for i in range(8):
    ax6[0][i].imshow(pool2_transpose[i][0])
    ax6[1][i].imshow(pool2_transpose[i + 8][0])
    ax6[2][i].imshow(pool2_transpose[i + 16][0])
    ax6[3][i].imshow(pool2_transpose[i + 24][0])
    ax6[4][i].imshow(pool2_transpose[i + 32][0])
    ax6[5][i].imshow(pool2_transpose[i + 40][0])
    ax6[6][i].imshow(pool2_transpose[i + 48][0])
    ax6[7][i].imshow(pool2_transpose[i + 56][0])
plt.title('Pool2 64x7x7')
plt.show()
