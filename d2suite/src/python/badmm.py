from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import unittest



def badmm_oneiter(Pi, Lambda, w1, w2, expD, epsilon = 1E-9):
    cPi = Pi * expD * tf.exp(-Lambda) + epsilon
    cPi = cPi * (w2 / tf.reduce_sum(cPi, 1, keep_dims = True)) 
    Pi = cPi * tf.exp(Lambda) + epsilon
    Pi = Pi * (w1 / tf.reduce_sum(Pi, 2, keep_dims = True))
    Lambda = Lambda + cPi - Pi
    return Pi, Lambda

def badmm(w1, w2, D, rho = 1, niter = 15):
    """
    The tensorflow implementation of Bregman ADMM method with auto-diff

    Args:
         w1, w2: two weight vectors (batch mode)         
         D: cost matrix
         rho: the parameter of ADMM
         niter: number of iterations
    
    Returns:
         Pi: the matching matrix between w1 and w2
    """
    n1 = w1.shape[0].value
    n2 = w2.shape[0].value
    assert(n2 == n1 or n1 == 1)
    expD = tf.expand_dims(tf.exp(D / rho), -1)
    d1 = w1.shape[-1].value
    d2 = w2.shape[-1].value
    w1 = tf.reshape(w1, [n1, d1, 1, 1])
    w2 = tf.reshape(w2, [n2, 1, d2, 1])
    Lambda = tf.constant(0., shape=[n2, d1, d2, 1])
    Pi = w1 * w2
    for i in range(niter):
        Pi, Lambda = badmm_oneiter(Pi, Lambda, w1, w2, expD)
    return tf.squeeze(Pi)


class TestBADMM(unittest.TestCase):
    def test_BADMM(self):
        w1 = tf.constant([[0.2, 0.3, 0.4, 0.1]]);
        w2 = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]]);
        D = tf.constant([[[0, 1, 2, 3], [1, 0, 1, 2],
                          [2, 1, 0, 1], [3, 2, 1, 0]]], dtype=tf.float32)
        Pi = badmm(w1, w2, D)
        Pi = tf.Print(Pi, [Pi], message="This is Pi: ")
        with tf.Session() as sess:
            sess.run(Pi)
            
if __name__ == "__main__":
    unittest.main()
