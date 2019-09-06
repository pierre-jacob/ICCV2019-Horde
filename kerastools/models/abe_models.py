#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Dense, Concatenate, Multiply, MaxPooling2D

from kerastools.layers import L2Normalisation
from kerastools.models.extractors import get_extractor, get_preprocess_method


def ABE(embedding_size=512, n_head=8):
    """
    Model from Kim et al. "Attention-based Ensemble for Deep Metric Learning", ECCV 2018.
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Wonsik_Kim_Attention-based_Ensemble_for_ECCV_2018_paper.pdf

    :param embedding_size: Size of the embedding. Must be a multiple of 'n_head' param.
    :param n_head: Number of attention block.
    :return: the model and the pre-processing method.
    """

    model = get_extractor('GoogleNet', end_layer=None)  # pre-build all layers and load weights.
    inputs = model.input
    attention_input = model.get_layer(name='pool3/3x3_s2').output

    def build_inception_block(stage, blk_input, filters, channel_axis=3, load_weights=True, multihead=False, prefix=""):
        (f11, (f21, f22), (f31, f32), f41) = filters
        blck_name = 'inception_' + stage
        if not load_weights:
            blck_name = 'att_' + blck_name

        layers = {blck_name + '/1x1': Conv2D(f11, (1, 1), padding='same', activation='relu', name=prefix+blck_name+'/1x1'),
                  blck_name + '/3x3_reduce': Conv2D(f21, (1, 1), padding='same', activation='relu', name=prefix+blck_name+'/3x3_reduce'),
                  blck_name + '/3x3': Conv2D(f22, (3, 3), padding='same', activation='relu', name=prefix+blck_name+'/3x3'),
                  blck_name + '/5x5_reduce': Conv2D(f31, (1, 1), padding='same', activation='relu', name=prefix+blck_name+'/5x5_reduce'),
                  blck_name + '/5x5': Conv2D(f32, (5, 5), padding='same', activation='relu', name=prefix+blck_name + '/5x5'),
                  blck_name + '/pool': MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=prefix+blck_name+'/pool'),
                  blck_name + '/pool_proj': Conv2D(f41, (1, 1), padding='same', activation='relu', name=prefix+blck_name+'/pool_proj'),
                  blck_name + '/output': Concatenate(axis=channel_axis, name=prefix+blck_name + '/output')}

        if multihead:
            X = []
            for inp in blk_input:
                inception_1x1 = layers[blck_name + '/1x1'](inp)
                inception_3x3_reduce = layers[blck_name + '/3x3_reduce'](inp)
                inception_3x3 = layers[blck_name + '/3x3'](inception_3x3_reduce)
                inception_5x5_reduce = layers[blck_name + '/5x5_reduce'](inp)
                inception_5x5 = layers[blck_name + '/5x5'](inception_5x5_reduce)
                inception_pool = layers[blck_name + '/pool'](inp)
                inception_pool_proj = layers[blck_name + '/pool_proj'](inception_pool)
                inception_output = layers[blck_name + '/output']([inception_1x1,
                                                                  inception_3x3,
                                                                  inception_5x5,
                                                                  inception_pool_proj])
                X.append(inception_output)
        else:
            inception_1x1 = layers[blck_name + '/1x1'](blk_input)
            inception_3x3_reduce = layers[blck_name + '/3x3_reduce'](blk_input)
            inception_3x3 = layers[blck_name + '/3x3'](inception_3x3_reduce)
            inception_5x5_reduce = layers[blck_name + '/5x5_reduce'](blk_input)
            inception_5x5 = layers[blck_name + '/5x5'](inception_5x5_reduce)
            inception_pool = layers[blck_name + '/pool'](blk_input)
            inception_pool_proj = layers[blck_name + '/pool_proj'](inception_pool)
            X = layers[blck_name + '/output']([inception_1x1,
                                               inception_3x3,
                                               inception_5x5,
                                               inception_pool_proj])

        if load_weights:
            for k in layers.keys():
                if k == 'inception_4b/output':  # handle a typo in the original model
                    tmp_k = 'inception_4b_output'
                else:
                    tmp_k = k
                layers[k].set_weights(model.get_layer(name=tmp_k).get_weights())

        return X

    # Shared Attention layers:
    pre_att = build_inception_block('4a', attention_input, (192, (96, 208), (16, 48), 64), load_weights=True, prefix='pre_')
    pre_att = build_inception_block('4b', pre_att, (160, (112, 224), (24, 64), 64), load_weights=True, prefix='pre_')
    pre_att = build_inception_block('4c', pre_att, (128, (128, 256), (24, 64), 64), load_weights=True, prefix='pre_')
    pre_att = build_inception_block('4d', pre_att, (112, (144, 288), (32, 64), 64), load_weights=True, prefix='pre_')
    pre_att = build_inception_block('4e', pre_att, (256, (160, 320), (32, 128), 128), load_weights=True, prefix='pre_')

    # Shared layers:
    MultiplyLayer = Multiply(name='Attention_weighting')
    Embedding = Dense(embedding_size // n_head, use_bias=False, name='Embedding')
    GAP = GlobalAveragePooling2D(name='GAP')
    L2 = L2Normalisation(name='L2')
    # Non-shared layers + last shared pre-built layers
    attention_outputs = []
    for m in range(n_head):
        att = Conv2D(filters=480,
                     kernel_size=(1, 1),
                     kernel_initializer='glorot_uniform',
                     activation='sigmoid',
                     name='att_' + str(m))(pre_att)
        x = MultiplyLayer([att, attention_input])
        attention_outputs.append(x)

    attention_outputs = build_inception_block('4a', attention_outputs, (192, (96, 208), (16, 48), 64), load_weights=True, multihead=True)
    attention_outputs = build_inception_block('4b', attention_outputs, (160, (112, 224), (24, 64), 64), load_weights=True, multihead=True)
    attention_outputs = build_inception_block('4c', attention_outputs, (128, (128, 256), (24, 64), 64), load_weights=True, multihead=True)
    attention_outputs = build_inception_block('4d', attention_outputs, (112, (144, 288), (32, 64), 64), load_weights=True, multihead=True)
    attention_outputs = build_inception_block('4e', attention_outputs, (256, (160, 320), (32, 128), 128), load_weights=True, multihead=True)
    for m in range(n_head):
        attention_outputs[m] = model.get_layer(name='pool4/3x3_s2')(attention_outputs[m])
    attention_outputs = build_inception_block('5a', attention_outputs, (256, (160, 320), (32, 128), 128), load_weights=True, multihead=True)
    attention_outputs = build_inception_block('5b', attention_outputs, (384, (192, 384), (48, 128), 128), load_weights=True, multihead=True)
    for m in range(n_head):
        attention_outputs[m] = GAP(attention_outputs[m])
        attention_outputs[m] = Embedding(attention_outputs[m])
        attention_outputs[m] = L2(attention_outputs[m])

    x = Concatenate(name='ABE'+str(n_head))(attention_outputs)

    return Model(inputs, x, name='ABE'+str(n_head)), get_preprocess_method('GoogleNet')
