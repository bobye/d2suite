from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from badmm import badmm
import unittest



def gwc_d2_model(w, V, mu_nbins, l2_penalty = 0.):
    """
    Args:
        w: [num_sample, num_bins]
        V: [num_sample, num_bins, dim]
    """
    num_sample = w.shape[0].value
    num_bins = V.shape[1].value
    dim = V.shape[-1].value

    V = V - tf.reduce_sum(tf.expand_dims(w, -1) * V, 1, keep_dims = True)

    L = tf.get_variable("L", shape=[mu_nbins, dim], dtype = tf.float32,
                        initializer = tf.constant_initializer(0.),
                        regularizer = tf.contrib.layers.l2_regularizer(l2_penalty))

    mu = tf.get_variable("mu", shape=[mu_nbins, dim], dtype = tf.float32,
                         initializer = tf.contrib.layers.variance_scaling_initializer(factor=10.))
    
    M = tf.expand_dims(V, 1) - tf.reshape(mu, [1, mu_nbins, 1, dim])

    D = tf.norm(M, ord = 2, axis = 3)
    meanD = 2.*tf.reduce_mean(D)
    #medianD = tf.contrib.distributions.percentile(D, 50.0)

    W = tf.get_variable("weights", shape=[mu_nbins], dtype = tf.float32,
                        initializer = tf.constant_initializer(1./mu_nbins))

    L1 = tf.reshape(L, [1, mu_nbins, 1, dim])

    E = tf.reduce_sum(L1 * M, -1)

    Pi = badmm(tf.expand_dims(W,0), w, D, rho = meanD)

    return tf.reduce_sum(Pi * E, [1, 2])

def gwc_hist_model(w, M, l2_penalty = 0.001, constr_penalty = 0.):
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
                        initializer = tf.constant_initializer(0.),
                        regularizer = tf.contrib.layers.l2_regularizer(l2_penalty))

    D = tf.norm(M, ord = 2, axis = 2)
    medianD = tf.contrib.distributions.percentile(D, 50.0)

    W = tf.get_variable("weights", shape=[num_bins], dtype = tf.float32,
                        initializer = tf.constant_initializer(1./num_bins))

    L1 = tf.expand_dims(L, 0)
    L2 = tf.expand_dims(L, 1)

    if constr_penalty > 0:
        constr = tf.reduce_mean(tf.nn.relu(tf.abs(tf.reduce_sum(L1 * M - L2 * M, 2)) - 2*D - 0.1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, constr_penalty * constr)

    E = tf.reduce_sum(L1 * M, 2)

    Pi = badmm(tf.expand_dims(W,0), w, D, rho = medianD)    
    

    return tf.reduce_sum(Pi * E, [1, 2])


def get_binary_losses(logit, label):
    """
    label: {-1., 1.}
    """
    return [tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)


def get_losses_gradients(logit, label, use_d2 = False):
    losses = get_binary_losses(logit, label)
    L = tf.get_variable("L")
    W = tf.get_variable("weights")

    if use_d2:
        mu = tf.get_variable("mu")
        dL, dW, dmu = tf.gradients(losses, xs = [L, W, mu])
        return tf.add_n(losses), [dL, dW, dmu]
    else:
        dL, dW = tf.gradients(losses, xs = [L, W])
        return tf.add_n(losses), [dL, dW]


def update_one_step(dLW, learning_rate = 0.01, step = None, use_d2 = False):
    L = tf.get_variable("L")
    W = tf.get_variable("weights")
    dL = dLW[0]
    dW = dLW[1]
    if use_d2:
        mu = tf.get_variable("mu")
        dmu = dLW[2]

    op1 = L.assign_sub(learning_rate * dL)
    if use_d2:
        with tf.control_dependencies([op1]):
            op1 = mu.assign_sub(learning_rate * dmu)

    op2 = W.assign(W / tf.exp(learning_rate * dW))
    with tf.control_dependencies([op2]):
        op2 = W.assign(W / tf.reduce_sum(W))

    with tf.control_dependencies([op1, op2]):
        if step is None:
            return tf.no_op()
        else:
            return tf.assign_add(step, 1)


def inference(logit):
    return tf.where(tf.less(logit, 0), tf.zeros_like(logit), tf.ones_like(logit))

def get_accuracy(logit, label):
    predicted_labels = inference(logit)
    return tf.reduce_sum(tf.cast(tf.equal(predicted_labels,label), tf.float32)) / label.shape[0].value

class TestBADMM(unittest.TestCase):
    def test_BADMM(self):
        w = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]], dtype=tf.float32);
        wtest = tf.constant([[0.1, 0.2, 0.4, 0.3], [0.6, 0.2, 0.1, 0.1]], dtype=tf.float32);
        M = tf.constant( [[[0,0], [0,-1], [-1,-1], [-1,0]],
                          [[0,1], [0,0],  [-1,1],  [-1,0]],
                          [[1,0], [1,-1], [0,0],   [0,-1]],
                          [[1,1], [1,0],  [0,1],   [0,0]]], dtype=tf.float32)
        label = tf.constant([1., -1.], dtype=tf.float32)

        with tf.variable_scope("test", reuse=tf.AUTO_REUSE):
            loss, dLW = get_losses_gradients(w, M, label)
            one_step = update_one_step(dLW, learning_rate = 0.1)
            predicted_labels = inference(wtest, M)

        loss = tf.Print(loss, [loss], message = "loss: ")
        predicted_labels = tf.Print(predicted_labels, [predicted_labels], message = "predicted: ")
        init = tf.global_variables_initializer()
            
        with tf.Session() as sess:
            sess.run(init)
            for i in range(1000):
                if (i+1) % 1 == 0:
                    sess.run(loss)
                sess.run(one_step)
            sess.run(predicted_labels)


if __name__ == "__main__":
    unittest.main()
