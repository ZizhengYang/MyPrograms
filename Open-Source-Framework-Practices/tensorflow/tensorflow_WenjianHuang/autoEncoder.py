import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
IMAGE_LENGTH = 28
INPUT = IMAGE_LENGTH * IMAGE_LENGTH
HIDDEN = 400
TRAINING_STEPS = 20
BATCH_SIZE = 128
DISPLAY_STEPS = 1
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "./model/"
'''

# Xaiver Initializer
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform(
        [fan_in, fan_out],
        minval=low,
        maxval=high,
        dtype=tf.float32
    )


class AdditiveGaussianNoiseAutocoder(object):
    def __init__(self,
                 n_input,                               # number of nodes of input layer and output layer
                 n_hidden,                              # number of nodes of hidden layer
                 transfer_function=tf.nn.softplus,      # activation function
                 optimizer=tf.train.AdamOptimizer(),    # optimizer
                 scale=0.1                              # coefficient of Gaussian Noise
                 ):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.weights = self._initializer_weights()
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(
            tf.add(
                tf.matmul(
                    self.x + scale * tf.random_normal(tf.shape(self.x), dtype=tf.float32),
                    self.weights['w1']
                ),
                self.weights['b1']
            )
        )
        self.reconstruction = tf.add(
            tf.matmul(
                self.hidden,
                self.weights['w2']
            ),
            self.weights['b2']
        )
        self.loss = .5 * tf.reduce_sum(tf.pow(
            tf.subtract(
                self.reconstruction,
                self.x
            ),
            2.0
        ))
        self.train_step = optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    # initialize all weights and bias
    def _initializer_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input]))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input]))
        return all_weights

    # calculate the loss value while training the model
    def partial_fit(self, X):
        loss, step = self.sess.run([self.loss, self.train_step], feed_dict={self.x: X, self.scale: self.training_scale})
        return loss

    # only calculate the loss value
    def calc_total_loss(self, X):
        return self.sess.run(self.loss, feed_dict={self.x: X, self.scale: self.training_scale})

    # the results in the hidden layer -> which indicate the advance features inside the data
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # use the results in the hidden layer as input to calculate the out layer -> recover the original data
    def generate(self, hidden=None):
        if hidden is None:
            hidden = tf.random_normal(tf.shape(self.weights['b1']))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # transform + generate
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# standardize the training data and test data
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# get random block from data
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    end_index = start_index + batch_size
    return data[start_index:end_index]


'''
def Train():
    mnist = input_data.read_data_sets('./dataBase/', one_hot=True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    NUM_SAMPLE = int(mnist.train.num_examples)

    AGN_autoencoder = AdditiveGaussianNoiseAutocoder(
        n_input=INPUT,
        n_hidden=HIDDEN,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE),
        scale=0.1
    )

    for r in range(TRAINING_STEPS):
        avg_loss = 0
        total_batch = int(NUM_SAMPLE / BATCH_SIZE)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(data=X_train, batch_size=BATCH_SIZE)
            loss = AGN_autoencoder.partial_fit(X=batch_xs)
            avg_loss = loss / NUM_SAMPLE * BATCH_SIZE
        if r % DISPLAY_STEPS == 0:
            print("Loss value in this round = ", end="")
            print(avg_loss)


if __name__ == '__main__':
    Train()
'''
if __name__ == "__main__":
    mnist = input_data.read_data_sets("MINIST_data/", one_hot=True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1

    autoencoder = AdditiveGaussianNoiseAutocoder(
        n_input=784,
        n_hidden=200,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        scale=0.01
    )

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
