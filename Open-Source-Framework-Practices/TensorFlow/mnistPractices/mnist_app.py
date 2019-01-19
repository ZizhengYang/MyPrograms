import os
import tensorflow as tf
import numpy as np
from PIL import Image
from mnistPractices import mnist_backward, mnist_forward
# import minst_forward, mnist_backward


def restore_model(testPicArr):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                return sess.run(preValue, feed_dict={x: testPicArr})
            else:
                print("No check point is found!")
                return -1;


def pre_pic(picName):
    img = Image.open(picName)
    reImg = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(reImg.convert('L'))
    threshold = 50  # Smaller: considered as black; Larger: consider as white

    for i in range(28):
        for j in range(28):
            img_arr[i][j] = 255 - img_arr[i][j]
            if img_arr[i][j] < threshold: img_arr[i][j] = 0
            else: img_arr[i][j] = 255

    num_arr = img_arr.reshape([1, 784])
    num_arr = num_arr.astype(np.float32)
    return np.multiply(num_arr, 1.0/255.0)


def application():

    testNum = input("Input the number of pictures: ")
    testNum = int(testNum)

    for i in range(testNum):
        testPic = input("The path of the test picture: ")

        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)

        print("The prediction number is: ", end="")
        print(preValue)


def main():
    application()


if __name__ == '__main__':
    main()
