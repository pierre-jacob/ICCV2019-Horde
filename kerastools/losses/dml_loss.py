#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from warnings import warn
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy


def __get_similarity_matrices(n_class, embeddings, sim='cosine', eps=1e-12):
    """Utility function to retrieve the similarity (cosine or l2) elements.

    Note that the batch must be built using M exclusive classes and N images per class, following this construction:
    im1_c1, im2_c1, ..., im1_c2, im2_c2, ... imN_cM
    Where imN_cM means "the N-th image for the M-th class".

    :param n_class: The number of exclusive classes.
    :param embeddings: The embeddings for all batch images
    :param sim: the similarity that is computed. Expect 'cosine' or 'l2' for now.
    :return: matrix of positive similarities [shape = MN x (M-1)] and matrix of negative similarities
     [shape = MN x (M-1)N]
    """
    s = tf.shape(embeddings)
    c = s[0] // n_class

    # Compute similarity matrix:
    if sim == 'cosine':
        embeddings /= tf.linalg.norm(embeddings, axis=1, keepdims=True) + eps
        sim_matrix = tf.matmul(embeddings, tf.transpose(embeddings))  # n_class * c x n_class * c
    elif sim == 'l2':
        n = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)
        sim_matrix = tf.matmul(embeddings, tf.transpose(embeddings))  # n_class * c x n_class * c
        sim_matrix = n + tf.transpose(n) - 2. * sim_matrix
    else:
        raise ValueError("Unknown similarity measure."
                         "Expect 'cosine' or 'l2' but got {}".format(sim))

    # Get only distances between queries and their respective positives:
    mask_c = tf.expand_dims(tf.expand_dims(tf.eye(n_class, dtype=tf.int64), axis=-1),
                            axis=-1)  # n_class x n_class x 1 x 1
    mask_p = tf.expand_dims(tf.expand_dims(tf.ones((c, c), dtype=tf.int64), axis=0), axis=0)  # 1 x 1 x c x c

    full_mask = mask_c * mask_p  # n_class x n_class x c x c
    full_mask = tf.transpose(full_mask, tf.stack([0, 2, 1, 3]))
    full_mask = tf.reshape(full_mask, tf.stack([n_class * c, n_class * c]))
    full_mask = full_mask - 2 * tf.eye(n_class * c, dtype=tf.int64)  # withdraw all similarities with themselves

    ind_p = tf.where(tf.equal(tf.cast(1, dtype=tf.int64), full_mask))  # n_class * c(c-1)
    ind_n = tf.where(tf.equal(tf.cast(0, dtype=tf.int64), full_mask))  # n_class * (n_class-1) * c^2

    sim_p = tf.gather_nd(sim_matrix, ind_p)
    sim_p = tf.reshape(sim_p, tf.stack([n_class*c, c-1]))

    sim_n = tf.gather_nd(sim_matrix, ind_n)
    sim_n = tf.reshape(sim_n, tf.stack([n_class, (n_class-1), c, c]))
    sim_n = tf.transpose(sim_n, tf.stack([0, 2, 1, 3]))
    sim_n = tf.reshape(sim_n, tf.stack([n_class*c, (n_class-1)*c]))

    return sim_p, sim_n


def make_multi_class_binomial_deviance(n_class, alpha=2., beta=0.5, cy=25., sim='cosine'):
    """ Builder for the Binomial Deviance loss function. Ref:
    https://papers.nips.cc/paper/6464-learning-deep-embeddings-with-histogram-loss.pdf

    Note that the batch must be built using M exclusive classes and N images per class, following this construction:
    im1_c1, im2_c1, ..., im1_c2, im2_c2, ... imN_cM
    Where imN_cM means "the N-th image for the M-th class".

    Arguments:
        n_class: Number of exclusive class within a batch.
        alpha: Parameter for binomial deviance, see the paper.
        beta: Parameter for binomial deviance, see the paper.
        cy: Parameter for binomial deviance, see the paper.
        sim: Computation of the similarity. Except 'cosine' or 'l2'.
    Returns:
        (Scalar) the loss.
    """
    if sim != 'cosine':
        raise ValueError("Binomial deviance expects cosine similarity only."
                         "This requirement may be changed in future version.")

    def binomial_deviance(y_true, y_pred):
        s = tf.shape(y_pred)
        c = s[0] // n_class
        sim_p, sim_n = __get_similarity_matrices(n_class, y_pred, sim=sim)

        # Compute the loss:
        loss_p = tf.reduce_sum(tf.reduce_mean(tf.log(1. + tf.exp(-alpha * (sim_p - beta))), axis=1))
        loss_n = tf.reduce_sum(tf.log(1. + tf.exp(alpha * cy * (sim_n - beta)))) / tf.cast(c * (n_class - 1), dtype=tf.float32)

        return loss_p + loss_n

    return binomial_deviance


def make_multi_class_triplet_loss(n_class, triplet_margin=0.1, epsilon=1e-12, sim='cosine'):
    """ Builder for an improved triplet loss function. Ref:
    http://openaccess.thecvf.com/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf

    Note that the batch must be built using M exclusive classes and N images per class.

    This is an improved version of the standard triplet loss. The improvements are:
        - Multi-class handling which leads to MN^2(N-1)(M-1) explored triplets
        - The loss is only average by the number of active triplets
        - The computation is fully paralleled

    The batch follows is constructed such as:
    im1_c1, im2_c1, ..., im1_c2, im2_c2, ... imN_cM
    Where imN_cM means "the N-th image for the M-th class".

    Arguments:
        n_class: Number of exclusive class within a batch.
        triplet_margin: The triplet margin. Standard value is 0.1.
        epsilon: small offset to avoid division by 0 when normalizing.
        sim: Computation of the similarity. Except 'cosine' or 'l2'.
    Returns:
        (Scalar) the loss.
    """
    def triplet_loss(y_true, y_pred):
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
        loss = tf.reduce_sum(loss) / (n_z + epsilon)
        return loss

    return triplet_loss


def make_multi_class_contrastive_loss(n_class, contrastive_margin=0.5, epsilon=1e-12, sim='cosine'):
    """ Builder for an improved contrastive loss function. Ref:
    http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf

    Note that the batch must be built using M exclusive classes and N images per class.

    This is an improved version of the standard contrastive loss. The improvements are:
        - Multi-class handling which leads to MN(N-1) + MN(M-1)N explored pairs
        - The loss is only average by the number of active pairs
        - The computation is fully paralleled

    The batch follows is constructed such as:
    im1_c1, im2_c1, ..., im1_c2, im2_c2, ... imN_cM
    Where imN_cM means "the N-th image for the M-th class".

    Arguments:
        n_class: Number of exclusive class within a batch.
        contrastive_margin: Contrastive loss margin. Standard value is 0.5.
        epsilon: small offset to avoid division by 0 when normalizing.
        sim: Computation of the similarity. Except 'cosine' or 'l2'.
    Returns:
        (Scalar) the loss.
    """
    def contrastive_loss(y_true, y_pred):
        sim_p, sim_n = __get_similarity_matrices(n_class, y_pred, sim=sim)

        if sim == 'cosine':
            loss_p = tf.reduce_mean(tf.square(sim_p - 1.)) / 2.
            loss_n = tf.nn.relu(sim_n - contrastive_margin)
        elif sim == 'l2':
            loss_p = tf.reduce_mean(sim_p) / 2.
            loss_n = tf.nn.relu(contrastive_margin - sim_n)
        else:
            raise ValueError("Unknown similarity measure."
                             "Expect 'cosine' or 'l2' but got {}".format(sim))

        n_z = tf.reduce_sum(tf.cast(tf.less(0., loss_n), tf.float32))
        loss_n = tf.reduce_sum(loss_n) / (2. * n_z + epsilon)

        return loss_p + loss_n

    return contrastive_loss


def make_multiclass_n_pairs_loss(n_class, n_pairs_margin=0., sigma=1., sim='cosine'):
    """ Builder for an improved N-pairs loss. Ref:
    https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf

    Note that the batch must be built using M exclusive classes and N images per class.

    This is an improved version of the standard N-pairs loss. The improvements are:
        - Multi-class handling which leads to MN^2(N-1)(M-1) explored triplets
        - The computation is fully paralleled

    The batch follows is constructed such as:
    im1_c1, im2_c1, ..., im1_c2, im2_c2, ... imN_cM
    Where imN_cM means "the N-th image for the M-th class".

    For now, this function cannot reproduces the results from the original paper. Indeed, it is also the same case for
    the N-pair loss implementation from Tensorflow contrib.

    Arguments:
        n_class: Number of exclusive class within a batch.
        n_pairs_margin: N-pairs loss margin. Standard value is 0.0.
        sigma: weight inside the exp. Similar to alpha in binomial deviance.
        sim: Computation of the similarity. Except 'cosine'.
    Returns:
        (Scalar) the loss.
    """
    warn("Before using this loss function, you have to know that I was never able to reproduce the results from "
         "the original paper. For example, Cars-196 datasets leads to 40.5 of recall@1 compared to ~71 in the paper.")

    if sim != 'cosine':
        raise ValueError("N-pair loss expects cosine similarity only."
                         "This requirement may be changed in future version.")

    def n_pairs_loss(y_true, y_pred):
        s = tf.shape(y_pred)
        c = s[0] // n_class
        sim_p, sim_n = __get_similarity_matrices(n_class, y_pred, sim=sim)

        sim_p = tf.reshape(sim_p, tf.stack([n_class*c, c-1, 1]))  # n_class*c x (c-1) x 1
        sim_n = tf.reshape(sim_n, tf.stack([n_class*c, 1, (n_class-1)*c]))  # n_class*c x 1 x (n_class-1)*c

        loss = tf.exp(sigma * (sim_n - sim_p + n_pairs_margin))  # n_class*c x c-1 x (n_class-1)*c

        loss = tf.reduce_mean(tf.log(tf.reduce_sum(loss, axis=-1) + 1.))

        return loss

    return n_pairs_loss


def make_multiclass_n_pairs_loss_v2():
    def n_pairs_loss(y_true, y_pred):
        s = tf.shape(y_pred)
        c = s[0] // 2

        # Compute similarity matrix:
        sim_matrix = tf.matmul(y_pred, tf.transpose(y_pred))  # n_class * c x n_class * c

        sim_matrix = tf.reshape(sim_matrix, tf.stack([c, 2, c, 2]))
        sim_matrix = tf.transpose(sim_matrix, tf.stack([1, 3, 0, 2]))

        labels = tf.eye(c)

        loss1 = categorical_crossentropy(labels, tf.cast(softmax(sim_matrix[0, 1], axis=-1), tf.float32))
        loss2 = categorical_crossentropy(labels, tf.cast(softmax(sim_matrix[1, 0], axis=-1), tf.float32))

        loss = tf.reduce_mean(loss1 + loss2) / 2.

        return loss

    return n_pairs_loss


def make_multi_class_multi_similarity_loss(n_class, alpha=2., beta=50., lamb=0.5, sim='cosine'):
    """ Builder for the Multi-similarity loss function. Ref:
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf

    Note that the batch must be built using M exclusive classes and N images per class, following this construction:
    im1_c1, im2_c1, ..., im1_c2, im2_c2, ... imN_cM
    Where imN_cM means "the N-th image for the M-th class".

    Arguments:
        n_class: Number of exclusive class within a batch.
        alpha: Parameter for MS loss, see the paper.
        beta: Parameter for MS loss, see the paper.
        lamb: Parameter for MS loss, see the paper.
        sim: Computation of the similarity. Except 'cosine' or 'l2'.
    Returns:
        (Scalar) the loss.
    """
    if sim != 'cosine':
        raise ValueError("Binomial deviance expects cosine similarity only."
                         "This requirement may be changed in future version.")

    def ms_loss(y_true, y_pred):
        sim_p, sim_n = __get_similarity_matrices(n_class, y_pred, sim=sim)

        # Compute the loss:
        loss_p = tf.log(1. + tf.reduce_sum(tf.exp(-alpha * (sim_p - lamb)), axis=-1)) / alpha
        loss_n = tf.log(1. + tf.reduce_sum(tf.exp(beta * (sim_n - lamb)), axis=-1)) / beta

        return tf.reduce_mean(loss_p + loss_n)

    return ms_loss
