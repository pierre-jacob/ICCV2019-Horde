#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Dense, Concatenate, Flatten, Input

from .abe_models import ABE
from ..layers import L2Normalisation
from ..layers.horde_layers import CKOP, PKOB
from .extractors import get_extractor, get_preprocess_method


def KOrderModel(extractor_name,
                embedding_sizes,
                high_order_dims,
                ho_trainable=False,
                end_layer=None):
    model = get_extractor(extractor_name, end_layer=end_layer)
    inputs = model.input
    x = model.output

    max_order = len(high_order_dims)
    output_list = [x]

    # Add all high-order approximation layers:
    for k, order_dim in enumerate(high_order_dims, start=2):
        x_ho = CKOP(output_dim=order_dim, name='CKOP_' + str(k), ho_trainable=ho_trainable)([x] * k)
        output_list.append(x_ho)

    # Add pooling and embedding layers:
    for k in range(len(output_list)):
        output_list[k] = GlobalAveragePooling2D(name='GAP_' + extractor_name + '_O' + str(k + 1))(output_list[k])
        if embedding_sizes[k] > 0:
            output_list[k] = Dense(embedding_sizes[k], use_bias=False)(output_list[k])
        output_list[k] = L2Normalisation(name='L2_' + extractor_name + '_O' + str(k + 1))(output_list[k])

    return Model(inputs=inputs, outputs=output_list, name=extractor_name + '_O' + str(max_order)), get_preprocess_method(extractor_name)


def CascadedKOrder(extractor_name,
                   embedding_sizes,
                   high_order_dims,
                   ho_trainable=True,
                   end_layer=None):
    model = get_extractor(extractor_name, end_layer=end_layer)
    inputs = model.input
    x = model.output

    max_order = len(high_order_dims)
    output_list = [x]

    # Add all high-order approximation layers:
    for k, order_dim in enumerate(high_order_dims, start=2):
        only_project_second = False if k == 2 else True
        x_ho = PKOB(order_dim,
                    only_project_second=only_project_second,
                    ho_trainable=ho_trainable)([output_list[-1], x])
        output_list.append(x_ho)

    # Add pooling and embedding layers:
    for k in range(len(output_list)):
        output_list[k] = GlobalAveragePooling2D(name='GAP_' + extractor_name + '_O' + str(k + 1))(output_list[k])

        if ho_trainable:
            output_list[k] = Dense(embedding_sizes[k],
                                   use_bias=False,
                                   name='Proj_' + extractor_name + '_O' + str(k + 1))(output_list[k])
        elif k == 0:
            output_list[k] = Dense(embedding_sizes[k],
                                   use_bias=False,
                                   name='Proj_' + extractor_name + '_O' + str(k + 1))(output_list[k])

        output_list[k] = L2Normalisation(name='L2_' + extractor_name + '_O' + str(k + 1))(output_list[k])

    return Model(inputs=inputs, outputs=output_list, name=extractor_name + '_O' + str(max_order)), get_preprocess_method(extractor_name)


def CascadedABE(embedding_size,
                high_order_dims,
                features_reduction=256,
                ho_trainable=True,
                n_head=8):
    model, preprocess_method = ABE(embedding_size[0], n_head=8)
    inp = model.input
    multi_head_out = [model.get_layer(name='inception_5b/output').get_output_at(k) for k in range(n_head)]
    concat = Concatenate()(multi_head_out)  # Nx H x W x n_ensemble*1024

    if features_reduction is not None:
        concat = Conv2D(filters=features_reduction,
                        kernel_size=(1, 1),
                        use_bias=False)(concat)

    output_list = [concat]

    # Add all high-order approximation layers:
    for k, order_dim in enumerate(high_order_dims, start=2):
        only_project_second = False if k == 2 else True
        x_ho = PKOB(order_dim,
                    only_project_second=only_project_second,
                    ho_trainable=ho_trainable)([output_list[-1], concat])
        output_list.append(x_ho)

    # Add pooling and embedding layers:
    for k in range(1, len(output_list)):
        output_list[k] = GlobalAveragePooling2D(name='GAP_O' + str(k + 1))(output_list[k])
        if ho_trainable:
            output_list[k] = Dense(embedding_size[k],
                                   use_bias=False,
                                   name='Proj_O' + str(k + 1))(output_list[k])
        output_list[k] = L2Normalisation(name='L2_O' + str(k + 1))(output_list[k])

    # Finally we replace the first order by the true model:
    output_list[0] = model.get_layer(name='ABE'+str(n_head)).output
    return Model(inp, output_list, name='ABE'+str(n_head)+'_O'+str(len(embedding_size))), preprocess_method
