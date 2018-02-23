from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import geodesic_wasserstein_classification as gwc
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('is_train', False,
                            """Whether to train """)

tf.app.flags.DEFINE_float('learning_rate', 1.,
                          """ the learning rate in sgd """)

tf.app.flags.DEFINE_integer('image_size', 20,
                            """ """)

tf.app.flags.DEFINE_integer('batch_size', 64, """ """)

def get_two_classes(dataset, a, b):
    select = np.where(dataset.labels[:,a] + dataset.labels[:,b] > 0)
    train_labels = np.squeeze(dataset.labels[select,a]*2-1)
    train_images = dataset.images[select]
    return train_images, train_labels

def get_M():
    M = np.zeros(shape = [ FLAGS.image_size *  FLAGS.image_size,  FLAGS.image_size *  FLAGS.image_size, 2], dtype = np.float32)
    for i in range(FLAGS.image_size*FLAGS.image_size):
        for j in range(FLAGS.image_size*FLAGS.image_size):
            xi = np.floor(i /FLAGS.image_size)
            yi = i % FLAGS.image_size
            xj = np.floor(i /FLAGS.image_size)
            yj = j % FLAGS.image_size
            M[i,j,0] = xi - xj
            M[i,j,1] = yi - yj
    return M


def get_crop_mask():
    mask = np.full(784, False)
    padding = FLAGS.image_size / 2
    for i in range(784):
        x = np.floor(i / 28)
        y = i % 28
        mask[i] = np.abs(x - 13.5) < padding and np.abs(y - 13.5) < padding
    return mask

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mask = get_crop_mask()
    train_images, train_labels = get_two_classes(mnist.train, 4, 9)
    train_images = train_images[:FLAGS.batch_size]
    train_images = train_images[:, mask]
    train_labels = train_labels[:FLAGS.batch_size]
    test_images, test_labels = get_two_classes(mnist.test, 4, 9)
    test_images = test_images[:, mask]

    lr = LogisticRegression()
    lr.fit(train_images, (train_labels+1) / 2)
    print('lr test accuracy: %f' % lr.score(test_images, (test_labels+1)/2))


    train_dataset = DataSet(train_images, train_labels, reshape=False)
    test_dataset = DataSet(test_images, test_labels, reshape=False)

    batch_size = FLAGS.batch_size
        
    dataM = get_M()
    

    w = tf.placeholder(shape=[batch_size, FLAGS.image_size * FLAGS.image_size], dtype = tf.float32)
    nw = w / tf.reduce_sum(w, 1, keep_dims=True)
    label = tf.placeholder(shape=[batch_size], dtype = tf.float32)
    M = tf.constant(get_M())
    global_step = tf.train.get_or_create_global_step()    

    with tf.variable_scope("gwc", reuse=tf.AUTO_REUSE):
        logit = gwc.gwc_hist_model(nw, M)
        loss, dLW = gwc.get_losses_gradients(logit, label)
        one_step = gwc.update_one_step(dLW, learning_rate = FLAGS.learning_rate,
                                       step = global_step)
        tf.summary.scalar('loss', loss)
        accuracy = gwc.get_accuracy(logit, label)

    loss = tf.Print(loss, [loss], message = "loss: ")
    init = tf.global_variables_initializer()


    saver = tf.train.Saver(tf.global_variables())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/mnist_logs')


    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 4
    config.gpu_options.allow_growth=True
    if FLAGS.is_train:
        ckpt = tf.train.get_checkpoint_state('/tmp/mnist_logs')
        with tf.Session(config=config) as sess:
            sess.run(init)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(1000):
                batch = train_dataset.next_batch(batch_size, shuffle=True)
                if (i+1) % 10 == 0:
                    summary, loss_v = sess.run([merged, loss],
                                           feed_dict = {w: batch[0], label: batch[1]})
                    writer.add_summary(summary, global_step.eval())
                
                sess.run(one_step, feed_dict = {w: batch[0], label: batch[1]})
                if (i+1) % 50 == 0:
                    saver.save(sess, '/tmp/mnist_logs/param',
                               global_step = global_step.eval(), write_meta_graph=False)
    else:
        ckpt = tf.train.get_checkpoint_state('/tmp/mnist_logs')
        with tf.Session(config=config) as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            acc_v = 0
            count = 0
            for i in tqdm(range(int(test_dataset.num_examples / batch_size))):
                test_batch = test_dataset.next_batch(batch_size, shuffle=False)
                acc_v += sess.run(accuracy, feed_dict = {w: test_batch[0], label: test_batch[1]})
                count += 1
                # print('batch #d: %f' % i, acc_v / count)
            acc_v /= count;
            print('test accuracy: %f' % acc_v)
