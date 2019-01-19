import os
import tensorflow as tf
os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_NODE = 784    # 28 * 28
OUTPUT_NODE = 10    # [0 1 2 3 4 5 6 7 8 9] one-hot representation
LAYER_NODE = 500    # number of layer nodes


def _get_weight(shape, regularizer):

    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def _get_bias(shape):

    return tf.Variable(tf.zeros(shape))


def forward(x, regularizer):

    w1 = _get_weight([INPUT_NODE, LAYER_NODE], regularizer)
    b1 = _get_bias([LAYER_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = _get_weight([LAYER_NODE, OUTPUT_NODE], regularizer)
    b2 = _get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2

    return y
