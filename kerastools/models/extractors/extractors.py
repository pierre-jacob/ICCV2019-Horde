#!/usr/bin/env python
# coding: utf-8
import configparser
from tensorflow.keras.models import Model, load_model

from ...utils.generic_utils import expanded_join
from ...layers.googlenet_layers import LRN


def GoogleNet(end_layer=None):
    config = configparser.ConfigParser()
    config.read(expanded_join('config.ini'))

    model_path = config['PROJECT_FOLDERS']['DATA_PATH']
    model = load_model(expanded_join(model_path, 'GoogleNet_notop.h5'), custom_objects={'LRN': LRN})

    if not end_layer is None:
        model = Model(inputs=model.input, outputs=model.get_layer(name=end_layer).output)

    return model


def BNInception(end_layer=None):
    config = configparser.ConfigParser()
    config.read(expanded_join('config.ini'))

    model_path = config['PROJECT_FOLDERS']['DATA_PATH']
    model = load_model(expanded_join(model_path, 'BN-Inception_notop.h5'))

    if not end_layer is None:
        model = Model(inputs=model.input, outputs=model.get_layer(name=end_layer).output)

    return model


def get_extractor(extractor_name, end_layer=None):
    return Extractors[extractor_name](end_layer=end_layer)


def get_preprocess_method(extractor_name):
    return Preprocess[extractor_name]


Extractors = {'GoogleNet': GoogleNet,
              'BNInception': BNInception}

Preprocess = {'GoogleNet': 'imagenet_centered',
              'BNInception': 'imagenet_centered'}
