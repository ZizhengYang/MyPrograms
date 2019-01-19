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
