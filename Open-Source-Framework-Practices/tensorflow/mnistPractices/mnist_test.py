import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as id
from mnistPractices import mnist_forward, mnist_backward
# import minst_forward, mnist_backward

TEST_INTERVAL_TIME = 5

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print("Accuracy: ", end="")
                print(accuracy_score)
                print("Round: ", end="")
                print(global_step)
            else:
                print("No check point is found!")
        time.sleep(TEST_INTERVAL_TIME)


def main():
    mnist = id.read_data_sets('./dataBase/', one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
