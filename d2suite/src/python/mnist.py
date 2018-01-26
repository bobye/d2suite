from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import geodesic_wasserstein_classification as gwc
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

from sklearn.linear_model import LogisticRegression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('is_train', False,
                            """Whether to train """)

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
    train_images = train_images[:128]
    train_labels = train_labels[:128]
    test_images, test_labels = get_two_classes(mnist.test, 2, 6)

    lr = LogisticRegression()
    lr.fit(train_images, (train_labels+1) / 2)
    print('lr test accuracy: %f' % lr.score(test_images, (test_labels+1)/2))


    train_dataset = DataSet(train_images, train_labels, reshape=False)
    test_dataset = DataSet(test_images, test_labels, reshape=False)

    if FLAGS.is_train:
        batch_size = 128
    else:
        batch_size = 128
        
    dataM = get_M()
    

    w = tf.placeholder(shape=[batch_size, 784], dtype = tf.float32)
    label = tf.placeholder(shape=[batch_size], dtype = tf.float32)
    M = tf.constant(get_M())

    with tf.variable_scope("gwc", reuse=tf.AUTO_REUSE):
        loss, dLW = gwc.get_losses_gradients(w, M, label)
        one_step = gwc.update_one_step(dLW, learning_rate = 1.)
        tf.summary.scalar('loss', loss)
        accuracy = gwc.get_accuracy(w, M, label)

    loss = tf.Print(loss, [loss], message = "loss: ")
    init = tf.global_variables_initializer()


    saver = tf.train.Saver(tf.global_variables())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/mnist_logs')


    if FLAGS.is_train:
        with tf.Session() as sess:
            sess.run(init)
            for i in range(1000):
                batch = train_dataset.next_batch(batch_size, shuffle=True)
                if (i+1) % 1 == 0:
                    summary, loss_v = sess.run([merged, loss],
                                           feed_dict = {w: batch[0], label: batch[1]})
                    writer.add_summary(summary, i)
                
                sess.run(one_step, feed_dict = {w: batch[0], label: batch[1]})
                if (i+1) % 10 == 0:
                    saver.save(sess, '/tmp/mnist_logs/param',
                               global_step = i, write_meta_graph=False)
    else:
        ckpt = tf.train.get_checkpoint_state('/tmp/mnist_logs')
        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            acc_v = 0
            count = 0
            for i in range(int(test_dataset.num_examples / batch_size)):
                test_batch = test_dataset.next_batch(batch_size, shuffle=False)
                acc_v += sess.run(accuracy, feed_dict = {w: test_batch[0], label: test_batch[1]})
                count += 1
                print('batch #d: %f' % i, acc_v / count)
            acc_v /= count;
            print('test accuracy: %f' % acc_v)
