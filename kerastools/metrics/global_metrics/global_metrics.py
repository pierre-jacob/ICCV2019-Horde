#!/usr/bin/env python
# coding: utf-8
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from abc import abstractmethod, ABC


_MAX_LIM = 250000000  # 1Go matrix maximum allowed (assuming float32).


class GlobalMetric(ABC):
    """ Global metric abstract class (all implemented metrics must inherit from it).

    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def compute_metric(self, predictions, labels):
        raise NotImplementedError("This function is currently not implemented.")

    @staticmethod
    def check_disrepancies(predictions: np.ndarray, labels: np.ndarray):
        # Number of samples:
        assert len(predictions) == len(labels), "'predictions' and 'labels' got a different number of samples. Got"\
                                                "(predictions){}=\={}(labels).".format(len(predictions), len(labels))
        # Labels and predictions formats:
        if labels.ndim == 1:
            unique_labels = list(set(labels))
        elif labels.ndim == 2:  # Expect a 0/1 sparse matrix of labels:
            labels = np.argmax(labels, axis=1)
            unique_labels = list(set(labels))
        else:
            raise ValueError("Could not understand 'labels' format."
                             "Expect a 1D or 2D ndarray, but got an array with shape {}.".format(np.shape(labels)))

        if predictions.ndim != 2:
            raise ValueError("Could not understand 'predictions' format."
                             "Expect a 2D ndarray, but got an array with shape {}.".format(np.shape(predictions)))


class KerasRecallAtK(GlobalMetric):
    """ Recall@K class as a global metrics.

    Arguments:
      ind_queries: indexes of all queries
      ind_collection: indexes of all images from the collection
      k_list: list of values of K
      similarity_measure: similarity measure. Should be 'cosine' or 'l2"
      queries_in_collection: set True if the queries are also in the collection for the search
    Returns:
      Instance of GlobalMetric to use in a GlobalMetricCallback.
    """
    def __init__(self, ind_queries,
                 ind_collection=None,
                 k_list: list = [1],
                 similarity_measure='cosine',
                 queries_in_collection=True):
        super().__init__('Recall@')
        self.k_list = k_list
        self.k_max = max(k_list)
        if len(ind_queries) == 0:
            raise ValueError('No query indexes given.')
        self.ind_queries = ind_queries
        self.ind_collection = ind_collection
        self.similarity_measure = similarity_measure

        offset = 1 if queries_in_collection else 0

        # Ranking computation graph:
        if similarity_measure == 'cosine':
            self.all_representations, self.input_labels, self.batch_representations,\
            self.batch_labels, self.ranking = _build_tf_cosine_similarity(max_rank=0, offset=offset)
        elif similarity_measure == 'l2':
            self.all_representations, self.input_labels, self.batch_representations,\
            self.batch_labels, self.ranking = _build_tf_l2_similarity(max_rank=0, offset=offset)
        else:
            raise NotImplementedError

        self.bin_ranking = K.cast(K.equal(self.ranking, self.batch_labels), K.floatx())

    def compute_metric(self, predictions: np.ndarray, labels: np.ndarray):
        self.check_disrepancies(predictions, labels)

        if self.ind_collection is None:
            collection = predictions.transpose()
            self.ind_collection = np.arange(len(predictions), dtype=np.int32)
        else:
            collection = predictions[self.ind_collection, :].transpose()

        # We can compute recall@K
        batch = int(np.ceil(_MAX_LIM / float(predictions.shape[0])))
        print('Computing {} steps.'.format(np.ceil(len(self.ind_queries) / batch)))

        sess = tf.get_default_session()
        if sess is None:
            sess = tf.Session()

        b = 0
        retrieved = np.zeros((len(self.k_list), 2), dtype=np.float32)
        retrieved[:, 0] = self.k_list

        while b < len(self.ind_queries):
            N = min(batch, len(self.ind_queries) - b)

            rnk = sess.run(self.bin_ranking,
                           feed_dict={self.all_representations: collection,
                                      self.input_labels: labels[self.ind_collection],
                                      self.batch_representations: predictions[self.ind_queries[b:b + N], :],
                                      self.batch_labels: labels[self.ind_queries[b:b + N], None]})
            b += N
            for i, k in enumerate(self.k_list):
                retrieved[i, 1] += np.sum(np.float32(np.max(rnk[:, 0:k], axis=1)))

        retrieved[:, 1] = (retrieved[:, 1] * 100) / float(len(self.ind_queries))

        return retrieved


def _build_tf_cosine_similarity(max_rank=0, offset=1, eps=1e-12):
    # We build the graph (See utils.generic_utils.tf_recall_at_k for original implementation):
    tf_db = K.placeholder(ndim=2, dtype=K.floatx())  # Where to find
    tf_labels = K.placeholder(ndim=1, dtype=K.floatx())  # and their labels

    tf_batch_query = K.placeholder(ndim=2, dtype=K.floatx())  # Used in case of memory issues
    batch_labels = K.placeholder(ndim=2, dtype=K.floatx())  # and their labels

    all_representations_T = K.expand_dims(tf_db, axis=0)  # 1 x D x N
    batch_representations = K.expand_dims(tf_batch_query, axis=0)  # 1 x n x D
    sim = K.batch_dot(batch_representations, all_representations_T)  # 1 x n x N
    sim = K.squeeze(sim, axis=0)  # n x N
    sim /= tf.linalg.norm(tf_batch_query, axis=1, keepdims=True) + eps
    sim /= tf.linalg.norm(tf_db, axis=0, keepdims=True) + eps

    if max_rank > 0:  # computing r@K or mAP@K
        index_ranking = tf.nn.top_k(sim, k=max_rank + offset).indices
    else:
        index_ranking = tf.contrib.framework.argsort(sim, axis=-1, direction='DESCENDING', stable=True)

    top_k = index_ranking[:, offset:]
    tf_ranking = tf.gather(tf_labels, top_k)

    return tf_db, tf_labels, tf_batch_query, batch_labels, tf_ranking


def _build_tf_l2_similarity(max_rank=0, offset=1):
    # We build the graph (See utils.generic_utils.tf_recall_at_k for original implementation):
    tf_db = K.placeholder(ndim=2, dtype=K.floatx())  # Where to find
    tf_labels = K.placeholder(ndim=1, dtype=K.floatx())  # and their labels

    tf_batch_query = K.placeholder(ndim=2, dtype=K.floatx())  # Used in case of memory issues
    batch_labels = K.placeholder(ndim=2, dtype=K.floatx())  # and their labels

    all_representations_T = K.expand_dims(tf_db, axis=0)  # 1 x D x N
    batch_representations = K.expand_dims(tf_batch_query, axis=0)  # 1 x n x D
    dist = -2. * K.batch_dot(batch_representations, all_representations_T)  # 1 x n x N
    dist = K.squeeze(dist, axis=0)  # n x N
    dist += K.sum(tf_batch_query * tf_batch_query, axis=1, keepdims=True)
    dist += K.sum(tf_db * tf_db, axis=0, keepdims=True)

    if max_rank > 0:  # computing r@K or mAP@K
        # top_k finds the k greatest entries and we want the lowest. Note that distance with itself will be last ranked
        dist = -dist
        index_ranking = tf.nn.top_k(dist, k=max_rank + offset).indices
    else:
        index_ranking = tf.contrib.framework.argsort(dist, axis=-1, direction='ASCENDING', stable=True)

    index_ranking = index_ranking[:, offset:]

    tf_ranking = tf.gather(tf_labels, index_ranking)

    return tf_db, tf_labels, tf_batch_query, batch_labels, tf_ranking
