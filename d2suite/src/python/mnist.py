from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import geodesic_wasserstein_classification as gwc
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

def get_two_classes(dataset, a, b):
    select = np.where(dataset.labels[:,a] + dataset.labels[:,b] > 0)
    train_labels = np.squeeze(dataset.labels[select,a]*2-1)
    train_images = dataset.images[select]
    return train_images, train_labels

def get_M():
    M = np.zeros(shape = [784, 784, 2], dtype = np.float32)
    for i in range(784):
        for j in range(784):
            xi = np.floor(i / 28)
            yi = i % 28
            xj = np.floor(i / 28)
            yj = j % 28
            M[i,j,0] = xi - xj
            M[i,j,1] = yi - yj
    return M

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train_images, train_labels = get_two_classes(mnist.train, 2, 6)
    
    test_images, test_labels = get_two_classes(mnist.test, 2, 6)

    train_dataset = DataSet(train_images, train_labels, reshape=False)
    test_dataset = DataSet(test_images, test_labels, reshape=False)

    batch_size = 4
    batch = train_dataset.next_batch(batch_size, shuffle=True)
    dataM = get_M()
    

    w = tf.placeholder(shape=[batch_size, 784], dtype = tf.float32)
    label = tf.placeholder(shape=[batch_size], dtype = tf.float32)
    M = tf.constant(get_M())

    with tf.variable_scope("gwc", reuse=tf.AUTO_REUSE):
        loss, dLW = gwc.get_losses_gradients(w, M, label)
        one_step = gwc.update_one_step(dLW, learning_rate = 0.1)
    
    loss = tf.Print(loss, [loss], message = "loss: ")
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            if (i+1) % 1 == 0:
                sess.run(loss, feed_dict = {w: batch[0], label: batch[1]})
            sess.run(one_step, feed_dict = {w: batch[0], label: batch[1]})
