#!/usr/bin/env python
# coding: utf-8
from .dml_loss import *


def get_dml_loss(name, n_class, sim='cosine'):
    if name == 'contrastive':
        return make_multi_class_contrastive_loss(n_class=n_class, contrastive_margin=0.5, sim=sim)
    elif name == 'triplet':
        return make_multi_class_triplet_loss(n_class=n_class, triplet_margin=0.1, sim=sim)
    elif name == 'binomial':
        return make_multi_class_binomial_deviance(n_class=n_class, sim=sim)
    elif name == "multi-similarity":
        return make_multi_class_multi_similarity_loss(n_class=n_class, sim=sim)
    else:
        raise ValueError("Unknown loss function. Expected 'contrastive', 'triplet', 'binomial' or 'multi-similarity'"
                         " but got {}".format(name))
