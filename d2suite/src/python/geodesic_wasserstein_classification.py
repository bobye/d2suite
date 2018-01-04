from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from badmm import badmm
import unittest



def gwc_model(w, M, l2_penalty = 0.001):
    """
    Args:
         w: [num_sample, num_bins] and tf.reduce_sum(w, 1) == tf.constant(1., [num_sample])
         M: [num_bins, num_bins, dim]
    """
    num_samples = w.shape[0].value
    num_bins = w.shape[1].value
    dim = M.shape[-1].value
    Mshape = M.get_shape()    
    assert(Mshape[0].value == num_bins and Mshape[1].value == num_bins and len(Mshape) == 3)

    L = tf.get_variable("L", shape=[num_bins, dim], dtype = tf.float32,
                        initializer = tf.random_normal_initializer(stddev=0.1),
                        regularizer = tf.contrib.layers.l2_regularizer(l2_penalty))

    D = tf.norm(M, ord = 2, axis = 2)

    W = tf.get_variable("weights", shape=[num_bins], dtype = tf.float32,
                        initializer = tf.constant_initializer(1./num_bins))

    E = tf.reduce_sum(tf.expand_dims(L, 0) * M, 2)

    medianD = tf.contrib.distributions.percentile(D, 50.0)
    
    Pi = badmm(tf.expand_dims(W,0), w, D, rho = medianD)

    return tf.reduce_sum(Pi * E, [1, 2])


def get_binary_losses(w, M, label):
    """
    label: {-1., 1.}
    """
    return [tf.reduce_mean(tf.sigmoid(gwc_model(w, M) * label))] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)


def get_losses_gradients(w, M, label):
    losses = get_binary_losses(w, M, label)
    L = tf.get_variable("L")
    W = tf.get_variable("weights")
    dL, dW = tf.gradients(losses, xs = [L, W])
    return tf.add_n(losses), [dL, dW]
    

class TestBADMM(unittest.TestCase):
    def test_BADMM(self):
        w = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]], dtype=tf.float32);
        M = tf.constant( [[[0,0], [0,-1], [-1,-1], [-1,0]],
                          [[0,1], [0,0],  [-1,1],  [-1,0]],
                          [[1,0], [1,-1], [0,0],   [0,-1]],
                          [[1,1], [1,0],  [0,1],   [0,0]]], dtype=tf.float32)
        label = tf.constant([1., -1.], dtype=tf.float32)

        with tf.variable_scope("test", reuse=tf.AUTO_REUSE):
            loss, dLW = get_losses_gradients(w, M, label)

        loss = tf.Print(loss, [loss], message = "loss: ")
        
        init = tf.global_variables_initializer()
            
        with tf.Session() as sess:
            sess.run(init)
            sess.run([loss, dLW])


if __name__ == "__main__":
    unittest.main()
