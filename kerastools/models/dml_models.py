#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D

from ..layers import L2Normalisation
from .extractors import get_extractor, get_preprocess_method


def Baseline(extractor_name,
             embedding_size,
             end_layer=None):
    model = get_extractor(extractor_name, end_layer=end_layer)
    inputs = model.input
    x = model.output

    x = Conv2D(embedding_size, (1, 1), use_bias=False, name='Embedding')(x)
    x = GlobalAveragePooling2D(name='GAP')(x)
    x = L2Normalisation(name='L2')(x)

    return Model(inputs=inputs, outputs=x, name=extractor_name), get_preprocess_method(extractor_name)
