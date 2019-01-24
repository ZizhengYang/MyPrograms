import cifar10, cifar10_input
# cifar10.maybe_download_and_extract()

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

max_steps = 6000
batch_size = 128
data_dir = './cifar10_data/cifar-10-batches-bin/'
MOVING_AVERAGE_DECAY = 0.99
MODEL_NAME = "cifar_model_cnn"
MODEL_SAVE_PATH = "./cnn_model2/"
sess = tf.InteractiveSession()


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weight_loss")
        tf.add_to_collection('losses', weight_loss)
    return var


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        name='loss-per-example'
    )
    loss_mean = tf.reduce_mean(loss, name='loss')
    tf.add_to_collection('losses', loss_mean)
    return tf.add_n(tf.get_collection('losses'))


images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [batch_size])

weight1 = variable_with_weight_loss([5, 5, 3, 64], 5e-2, w1=0.0)
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding="SAME") + bias1)
pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

weight2 = variable_with_weight_loss([5, 5, 64, 64], 5e-2, w1=0.0)
bias2 = tf.Variable(tf.constant(0.0, shape=[64]))
conv2 = tf.nn.relu(tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding="SAME") + bias2)
pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")

h_pool2_flat = tf.reshape(pool2, [batch_size, -1])
dim = h_pool2_flat.get_shape()[1].value
weight3 = variable_with_weight_loss([dim, 384], stddev=0.04, w1=0.004)
# print(h_pool2_flat)     # Tensor("Reshape_2:0", shape=(128, 2304), dtype=float32)
# print(dim)              # 2304
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(h_pool2_flat, weight3) + bias3)

keep_prob = tf.placeholder(tf.float32)
local3_drop = tf.nn.dropout(local3, keep_prob)

weight4 = variable_with_weight_loss([384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3_drop, weight4) + bias4)

weight5 = variable_with_weight_loss([192, 10], stddev=0.04, w1=0.004)
bias5 = tf.Variable(tf.constant(0.1, shape=[10]))
local5 = tf.nn.relu(tf.matmul(local4, weight5) + bias5)

loss = loss(local5, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(local5, label_holder, 1)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

tf.train.start_queue_runners()

Loss = []

for i in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if i % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        print("Step: ", end="")
        print(i, end="")
        print(" <==> Loss value is: ", end="")
        print(loss)

num_examples = 10000
import math


saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cross Loss')
plt.plot(Loss)
plt.grid()
plt.show()
