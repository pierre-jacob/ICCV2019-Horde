#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf

from ...losses.dml_loss import __get_similarity_matrices


def make_multi_class_count_triplet(n_class, triplet_margin=0.1, sim='cosine'):
    """ Builder for a multi-class active triplet counter.

    Note that the batch must be built using M exclusive classes and N images per class, following this construction:
    im1_c1, im2_c1, ..., im1_c2, im2_c2, ... imN_cM
    Where imN_cM means "the N-th image for the M-th class".

    Arguments:
        n_class: Number of exclusive class within a batch.
        triplet_margin: triplet loss margin.
        sim: similarity measure, should be 'cosine' or 'l2'.
    Returns:
        (Scalar) the metric.
    """
    def c_t(y_true, y_pred):
        s = tf.shape(y_pred)
        c = s[0] // n_class
        sim_p, sim_n = __get_similarity_matrices(n_class, y_pred, sim=sim)

        sim_p = tf.reshape(sim_p, tf.stack([n_class*c, c-1, 1]))  # n_class*c x (c-1) x 1
        sim_n = tf.reshape(sim_n, tf.stack([n_class*c, 1, (n_class-1)*c]))  # n_class*c x 1 x (n_class-1)*c

        if sim == 'cosine':
            loss = tf.nn.relu(sim_n - sim_p + triplet_margin)  # n_class*c x (c-1) x (n_class-1)*c
        elif sim == 'l2':
            loss = tf.nn.relu(sim_p - sim_n + triplet_margin)  # n_class*c x (c-1) x (n_class-1)*c
        else:
            raise ValueError("Unknown similarity measure."
                             "Expect 'cosine' or 'l2' but got {}".format(sim))

        n_z = tf.reduce_sum(tf.cast(tf.less(0., loss), tf.float32))

        return n_z

    return c_t


def make_multi_class_count_contrastive_pairs(n_class, contrastive_margin=0.5, sim='cosine'):
    """ Builder for a multi-class active triplet counter.

    Note that the batch must be built using M exclusive classes and N images per class, following this construction:
    im1_c1, im2_c1, ..., im1_c2, im2_c2, ... imN_cM
    Where imN_cM means "the N-th image for the M-th class".

    Arguments:
        n_class: Number of exclusive class within a batch.
        contrastive_margin: contrastive margin.
        sim: similarity measure, should be 'cosine' or 'l2'.
    Returns:
        (Scalar) the metric.
    """
    def c_c(y_true, y_pred):
        s = tf.shape(y_pred)
        c = s[0] // n_class
        sim_p, sim_n = __get_similarity_matrices(n_class, y_pred, sim=sim)

        if sim == 'cosine':
            loss_n = tf.nn.relu(sim_n - contrastive_margin)
        elif sim == 'l2':
            loss_n = tf.nn.relu(contrastive_margin - sim_n)
        else:
            raise ValueError("Unknown similarity measure."
                             "Expect 'cosine' or 'l2' but got {}".format(sim))

        n_z = tf.reduce_sum(tf.cast(tf.less(0., loss_n), tf.float32)) + tf.cast(s[0]*(c-1), tf.float32)

        return n_z

    return c_c
