import os
import tensorflow as tf
from tensorflow.python.training.saver import Saver

from mnistPractices import mnist_forward
# import mnist_forward
from tensorflow.examples.tutorials.mnist import input_data as id

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model_a"


def backward(mnist):

    TOTAL_BATCHES = mnist.train.num_examples // BATCH_SIZE

    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])

    y = mnist_forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        TOTAL_BATCHES,
        LEARNING_RATE_DECAY,
        staircase=True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            train, loss_value, step_value = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            learning = sess.run(learning_rate)
            if i % 1000 == 0:
                print("After ", end="")
                print(step_value-1, end="")
                print(" training step[s], loss of the training batch is: ", end="")
                print(loss_value, end="")
                print(", with the learning rate of ", end="")
                print(learning)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = id.read_data_sets('./dataBase/', one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()
