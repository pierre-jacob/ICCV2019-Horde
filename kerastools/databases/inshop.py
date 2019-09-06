#!/usr/bin/env python
# coding: utf-8
import numpy as np
from warnings import warn

from .databases import RetrievalDb
from ..utils.generic_utils import expanded_join


class InShop(RetrievalDb):
    """ In-Shop Cloth Retrieval dataset wrapper. Refs:
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf

    Arguments:

    Returns:
      Instance of InShop to get images and labels from train/val/test sets for DML tasks.
    """
    def __init__(self):
        super(InShop, self).__init__(name='INSHOP', queries_in_collection=False)
        self._parse_annotations()

    def get_training_set(self, **kwargs):
        return self.im_train, self.y_train

    def get_validation_set(self, **kwargs):
        super(InShop).get_validation_set(**kwargs)

    def get_testing_set(self, **kwargs):
        im = np.concatenate([self.im_query, self.im_gallery])
        lbl = np.concatenate([self.y_query, self.y_gallery])
        return im, lbl

    def _parse_annotations(self):
        """ Automatic parsing of available files to get train/test images and labels for the classification task. Note
        that the split in DML is different from the classification one.

        :param db: either 'train' or 'test'.
        :return: bounding boxes, labels and image paths.
        """
        with open(expanded_join(self.root_path, 'list_eval_partition.txt')) as f:
            f.readline()
            f.readline()  # withdraw header

            y_train = []
            y_query = []
            y_gallery = []

            im_train = []
            im_query = []
            im_gallery = []

            for l in f:
                data = l.split(' ')
                data = [x for x in data if x]

                if 'train' in data[-2]:
                    im_train.append(expanded_join(self.root_path, data[0]))
                    y_train.append(data[1])

                elif 'query' in data[-2]:
                    im_query.append(expanded_join(self.root_path, data[0]))
                    y_query.append(data[1])

                elif 'gallery' in data[-2]:
                    im_gallery.append(expanded_join(self.root_path, data[0]))
                    y_gallery.append(data[1])

                else:
                    print("Should be an error: got {}".format(data))

        self.im_train = np.array(im_train, dtype=str)
        self.im_query = np.array(im_query, dtype=str)
        self.im_gallery = np.array(im_gallery, dtype=str)
        self.y_train = np.array(y_train, dtype=str)
        self.y_query = np.array(y_query, dtype=str)
        self.y_gallery = np.array(y_gallery, dtype=str)

    @staticmethod
    def get_usual_retrieval_rank():
        return [1, 10, 20, 30, 40, 50]

    def get_queries_idx(self, db_set):
        """ Get the set of query images from which metrics are evaluated.

        :param db_set: string containing either 'train', 'training', 'validation', 'val', 'testing' or 'test'.
        :return: a nd-array of query indexes.
        """
        if db_set.lower() == 'train' or db_set.lower() == 'training':
            warn("User Warning: this function was used to collect queries' indexes from the training set."
                 "All indexes were return, do not take into account 'queries_in_collection' param.")
            return np.arange(len(self.im_train), dtype=np.int32)
        elif db_set.lower() == 'validation' or db_set.lower() == 'val':
            raise ValueError('There is no validation set for {}.'.format(self.name))
        elif db_set.lower() == 'testing' or db_set.lower() == 'test':
            return np.arange(len(self.im_query), dtype=np.int32)
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
            warn("User Warning: this function was used to collect queries' indexes from the training set."
                 "All indexes were return, do not take into account 'queries_in_collection' param.")
            return np.arange(len(self.im_train), dtype=np.int32)
        elif db_set.lower() == 'validation' or db_set.lower() == 'val':
            raise ValueError('There is no validation set for {}.'.format(self.name))
        elif db_set.lower() == 'testing' or db_set.lower() == 'test':
            return np.arange(start=len(self.im_query),stop=len(self.im_query)+len(self.im_gallery), dtype=np.int32)
        else:
            raise ValueError("'db_set' unrecognized."
                             "Expected 'train', 'training', 'validation', 'val', 'testing', 'test'"
                             "Got {}".format(db_set))
