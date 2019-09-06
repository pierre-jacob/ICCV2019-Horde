#!/usr/bin/env python
# coding: utf-8
import numpy as np

from .databases import RetrievalDb
from ..utils.generic_utils import expanded_join


class StanfordOnlineProducts(RetrievalDb):
    """ Stanford Online Products dataset wrapper. Refs:
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S17-11.pdf

    Arguments:

    Returns:
      Instance of StanfordOnlineProducts to get images and labels from train/val/test sets for DML tasks.
    """
    def __init__(self):
        super(StanfordOnlineProducts, self).__init__(name='STANFORD_ONLINE_PRODUCTS', queries_in_collection=True)
        self.train_images, self.train_labels = self._make_images_and_labels("Ebay_train.txt", True)
        self.test_images, self.test_labels = self._make_images_and_labels("Ebay_test.txt", True)

    def get_training_set(self, **args):
        return self.train_images, self.train_labels

    def get_validation_set(self, **args):
        super(StanfordOnlineProducts).get_validation_set(**args)

    def get_testing_set(self, **args):
        return self.test_images, self.test_labels

    def _make_images_and_labels(self, filename, is_retrieval):
        """ Automatic parsing of available files to get train/test images and labels for the classification task. Note
        that the split in DML is different from the classification one.

        :param db: either 'train' or 'test'.
        :return: bounding boxes, labels and image paths.
        """
        labels = []
        images = []
        j = 1 if is_retrieval else 2

        with open(expanded_join(self.root_path, filename)) as f:
            for i, l in enumerate(f):
                if i != 0:
                    data = l.split(' ')
                    labels.append(np.int32(data[j]) - 1)
                    images.append(expanded_join(self.root_path, data[3][0:-1]))

        images = np.array(images, dtype=np.str)
        labels = np.array(labels, dtype=np.int32)

        return images, labels

    @staticmethod
    def get_usual_retrieval_rank():
        return [1, 10, 100, 1000]

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