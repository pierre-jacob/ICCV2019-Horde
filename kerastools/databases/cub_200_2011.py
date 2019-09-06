#!/usr/bin/env python
# coding: utf-8
import numpy as np

from ..utils.generic_utils import expanded_join
from .databases import RetrievalDb


class Cub200Ret(RetrievalDb):
    """ Calltech Birs-200-2011 dataset wrapper. Refs:
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

    Arguments:

    Returns:
      Instance of Cub200Ret to get images and labels from train/val/test sets for DML tasks.
    """
    def __init__(self):
        super(Cub200Ret, self).__init__(name='RET_CUB_200_2011', queries_in_collection=True)
        cls_train_x1, cls_train_x2, cls_train_y1, \
        cls_train_y2, cls_train_labels, cls_train_images = self._parse_annotations('train')
        cls_test_x1, cls_test_x2, cls_test_y1, \
        cls_test_y2, cls_test_labels, cls_test_images = self._parse_annotations('test')

        labels = np.concatenate((cls_train_labels, cls_test_labels))
        images = np.concatenate((cls_train_images, cls_test_images))
        train_ind = np.where(labels <= 100)[0]
        test_ind = np.where(labels > 100)[0]
        x1 = np.concatenate((cls_train_x1, cls_test_x1))
        x2 = np.concatenate((cls_train_x2, cls_test_x2))
        y1 = np.concatenate((cls_train_y1, cls_test_y1))
        y2 = np.concatenate((cls_train_y2, cls_test_y2))

        self.train_images = images[train_ind]
        self.train_labels = labels[train_ind]
        self.train_x1 = x1[train_ind]
        self.train_x2 = x2[train_ind]
        self.train_y1 = y1[train_ind]
        self.train_y2 = y2[train_ind]

        self.test_images = images[test_ind]
        self.test_labels = labels[test_ind]
        self.test_x1 = x1[test_ind]
        self.test_x2 = x2[test_ind]
        self.test_y1 = y1[test_ind]
        self.test_y2 = y2[test_ind]

    def get_training_set(self, **kwargs):
        return self.train_images, self.train_labels

    def get_validation_set(self, **kwargs):
        super(Cub200Ret).get_validation_set(**kwargs)

    def get_testing_set(self, **kwargs):
        return self.test_images, self.test_labels

    @staticmethod
    def get_usual_retrieval_rank():
        return [1, 2, 4, 8, 16, 32]

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

    def get_bounding_boxes(self, db_set):
        """ Get the bounding boxes for each images.

        :param db_set: string containing either 'train', 'training', 'validation', 'val', 'testing' or 'test'.
        :return: a tuple of nd-array such that (left, right, top, bottom).
        """
        if db_set.lower() == 'training' or db_set.lower() == 'train':
            return self.train_x1, self.train_x2, self.train_y1, self.train_y2
        elif db_set.lower() == 'validation' or db_set.lower() == 'val':
            raise ValueError('There is no validation set for {}.'.format(self.name))
        elif db_set.lower() == 'testing' or db_set.lower() == 'test':
            return self.test_x1, self.test_x2, self.test_y1, self.test_y2
        else:
            raise ValueError("'db_set' unrecognized."
                             "Expected 'train', 'training', 'validation', 'val', 'testing' or 'test'"
                             "Got {}".format(db_set))

    def _parse_annotations(self, db):
        """ Automatic parsing of available files to get train/test images and labels for the classification task. Note
        that the split in DML is different from the classification one.

        :param db: either 'train' or 'test'.
        :return: bounding boxes, labels and image paths.
        """
        images = []
        labels = []
        train_ind = []
        test_ind = []
        x1 = []
        x2 = []
        y1 = []
        y2 = []

        with open(expanded_join(self.root_path, 'images.txt')) as f:
            for l in f:
                split_l = l.split(' ')
                images.append(expanded_join(self.root_path, 'images', split_l[1][:-1]))

        with open(expanded_join(self.root_path, 'train_test_split.txt')) as f:
            for l in f:
                split_l = l.split(' ')
                if split_l[1][:-1] == '1':
                    train_ind.append(int(split_l[0]) - 1)
                else:
                    test_ind.append(int(split_l[0]) - 1)

        with open(expanded_join(self.root_path, 'image_class_labels.txt')) as f:
            for l in f:
                labels.append(int(l.split(' ')[1]))

        with open(expanded_join(self.root_path, 'bounding_boxes.txt')) as f:
            for l in f:
                split_l = l.split(' ')
                x1.append(float(split_l[1]))
                y1.append(float(split_l[2]))
                x2.append(float(split_l[3]) + x1[-1])
                y2.append(float(split_l[4][:-1]) + y1[-1])

        train_ind = np.array(train_ind, dtype=np.int32)
        test_ind = np.array(test_ind, dtype=np.int32)
        images = np.array(images)
        labels = np.array(labels, dtype=np.int32)
        x1 = np.array(x1, dtype=np.int32)
        y1 = np.array(y1, dtype=np.int32)
        x2 = np.array(x2, dtype=np.int32)
        y2 = np.array(y2, dtype=np.int32)

        if db == 'train':
            ind = train_ind
        elif db == 'test':
            ind = test_ind
        else:
            raise ValueError("'db' param waits for 'train' or 'test' but get %s" % db)

        return x1[ind], x2[ind], y1[ind], y2[ind], labels[ind], images[ind]
