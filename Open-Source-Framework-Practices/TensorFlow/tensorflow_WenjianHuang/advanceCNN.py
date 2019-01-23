import tensorflow as tf
import os
import numpy as np
import time
import tensorflow_WenjianHuang.cifar_dataBase.models.tutorials.image.cifar10.cifar10 as cifar10, \
    tensorflow_WenjianHuang.cifar_dataBase.models.tutorials.image.cifar10.cifar10_input as input

max_steps = 3000
batch_size = 128
data_dir = 'cifar_dataBase/tmp/cifar10_data/cifar-10-batches-bin'


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weight_loss")
        tf.add_to_collection('losses', weight_loss)
    return var


cifar10.maybe_download_and_extract()
images_train, labels_train = input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=batch_size
                                                    )
images_test, labels_test = input.inputs(eval_data=True,
                                        data_dir=data_dir,
                                        batch_size=batch_size
                                        )

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [batch_size])

weight1 = variable_with_weight_loss([5, 5, 3, 64], 5e-2, w1=0.0)
bias1 = tf.Variable(tf.constant(0.0, [64]))
conv1 = tf.nn.relu(tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding="SAME") + bias1)
pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

weight2 = variable_with_weight_loss([5, 5, 64, 64], 5e-2, w1=0.0)
bias2 = tf.Variable(tf.constant[0.0, [64]])
conv2 = tf.nn.relu(tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding="SAME") + bias2)
pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")

w_fc1 = tf.Variable(tf.float32, [7*7*64, 1024])
b_fc1 = tf.Variable(tf.zeros([1024]))
h_pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = tf.Variable(tf.float32, [1024, 10])
b_fc2 = tf.Variable(tf.zeros([10]))
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

global_step = tf.Variable(tf.int32, trainable=False)

'''
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    TOTAL_BATCHES,
    LEARNING_RATE_DECAY,
    staircase=True
)
'''

# ce = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])
# cem = tf.reduce_mean(ce)
# loss = cem + tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)
