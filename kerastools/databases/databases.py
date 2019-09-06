#!/usr/bin/env python
# coding: utf-8
import os
import configparser
from abc import abstractmethod, ABC

from ..utils.generic_utils import expanded_join


class Database(ABC):
    """ Abstract database class. Do not touch this one.
    Arguments:
     name: dataset name to be search in the config file.
    """
    def __init__(self, name):
        if name is not None:
            config = configparser.ConfigParser()
            config.read(expanded_join('config.ini'))

            self.root_path = config['DATABASES_PATHS'][name + '_PATH']
            if name + '_LOCAL_PATH' in config['DATABASES_PATHS'].keys():
                if os.path.exists(config['DATABASES_PATHS'][name + '_LOCAL_PATH']):
                    self.root_path = config['DATABASES_PATHS'][name + '_LOCAL_PATH']

            self.name = name

    @abstractmethod
    def get_training_set(self, **kwargs):
        """ This function prepare the training set by providing either a ndarray of images either a ndarray of paths but
        also the respective labels (classification or retrieval doesn't matter).

        This function should be overload for dataset with both classification and retrieval labels.
        """
        raise NotImplementedError("This database does not have a training set.")

    @abstractmethod
    def get_validation_set(self, **kwargs):
        """ This function prepare the validation set by providing either a ndarray of images either a ndarray of paths
        but also the respective labels (classification or retrieval doesn't matter).

        This function should be overload for dataset with both classification and retrieval labels.
        """
        raise NotImplementedError("This database does not have a validation set.")

    @abstractmethod
    def get_testing_set(self, **kwargs):
        """ This function prepare the testing set by providing either a ndarray of images either a ndarray of paths but
        also the respective labels (classification or retrieval doesn't matter).

        This function should be overload for dataset with both classification and retrieval labels.
        """
        raise NotImplementedError("This database does not have a testing set.")


class RetrievalDb(Database, ABC):
    """ Abstract class for retrieval datasets. It allows a unique representation and generic usage of these sets.

    Arguments:
     name: dataset name to be search in the config file.
     queries_in_collection: If the image queries are got from the image collection.
    """
    def __init__(self, name, queries_in_collection):
        super(RetrievalDb, self).__init__(name=name)
        self.queries_in_collection = queries_in_collection

    @staticmethod
    @abstractmethod
    def get_usual_retrieval_rank():
        raise NotImplementedError("No usual ranking available.")

    @abstractmethod
    def get_queries_idx(self, db_set):
        raise NotImplementedError("'get_queries_idx' function not implemented.")

    @abstractmethod
    def get_collection_idx(self, db_set):
        raise NotImplementedError("'get_collection_idx' function not implemented.")
