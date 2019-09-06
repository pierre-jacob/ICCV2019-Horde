#!/usr/bin/env python
# coding: utf-8
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

from .databases import RetrievalDb


class MnistRet(RetrievalDb):
    """ Mnist wrapper. Refs:

    Arguments:

    Returns:
      Instance of MnistRet to get images and labels from train/val/test sets for DML tasks.
    """
    def __init__(self, **kwargs):
        super(MnistRet, self).__init__(name=None, queries_in_collection=True)
        self.name = "RET_MNIST"
        (x_train, y_train), (x_test, y_test) = load_data()
        idx_train = np.where(y_train < 5)[0]
        idx_test = np.where(y_test < 5)[0]
        self.train_images = np.concatenate([x_train[idx_train], x_test[idx_test]], axis=0)
        self.train_labels = np.concatenate([y_train[idx_train], y_test[idx_test]], axis=0)

        idx_train = np.where(y_train >= 5)[0]
        idx_test = np.where(y_test >= 5)[0]
        self.test_images = np.concatenate([x_train[idx_train], x_test[idx_test]], axis=0)
        self.test_labels = np.concatenate([y_train[idx_train], y_test[idx_test]], axis=0)

        self.train_images = self.train_images[..., None]
        self.test_images = self.test_images[..., None]

    def get_training_set(self, **kwargs):
        return self.train_images, self.train_labels

    def get_validation_set(self, **kwargs):
        super(MnistRet).get_validation_set(**kwargs)

    def get_testing_set(self, **kwargs):
        return self.test_images, self.test_labels

    @staticmethod
    def get_usual_retrieval_rank():
        return [1, 2, 10]

    def get_queries_idx(self, db_set):
        """ Get the set of query images from which metrics are evaluated.

        :param db_set: string containing either 'train', 'training', 'validation', 'val', 'testing' or 'test'.
        :return: a nd-array of query indexes.
        """
        if db_set.lower() == 'train' or db_set.lower() == 'training':
            return np.arange(len(self.train_images), dtype=np.int32)
        elif db_set.lower() == 'validation' or db_set.lower() == 'val':
            raise ValueError('There is no validation set for {}.'.format(self.name))
        elif db_set.lower() == 'testing' or db_set.lower() == 'test':
            return np.arange(len(self.test_images), dtype=np.int32)
        else:
            raise ValueError("'db_set' unrecognized."
                             "Expected 'train', 'training', 'validation', 'val', 'testing', 'test'"
                             "Got {}".format(db_set))

    def get_collection_idx(self, db_set):
        """ Get the set of collection images for retrieval tasks.

        :param db_set: string containing either 'train', 'training', 'validation', 'val', 'testing' or 'test'.
        :return: a nd-array of the collection indexes.
        """
        if db_set.lower() == 'train' or db_set.lower() == 'training':
            return np.arange(len(self.train_images), dtype=np.int32)
        elif db_set.lower() == 'validation' or db_set.lower() == 'val':
            raise ValueError('There is no validation set for {}.'.format(self.name))
        elif db_set.lower() == 'testing' or db_set.lower() == 'test':
            return np.arange(len(self.test_images), dtype=np.int32)
        else:
            raise ValueError("'db_set' unrecognized."
                             "Expected 'train', 'training', 'validation', 'val', 'testing', 'test'"
                             "Got {}".format(db_set))
