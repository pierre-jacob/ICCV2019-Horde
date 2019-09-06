#!/usr/bin/env python
# coding: utf-8

from .dml_metrics import *


def get_dml_metric(name, n_class, sim='cosine'):
    if name == 'contrastive':
        return make_multi_class_count_contrastive_pairs(n_class=n_class, contrastive_margin=0.5, sim=sim)
    elif name == 'triplet':
        return make_multi_class_count_triplet(n_class=n_class, triplet_margin=0.1, sim=sim)
    elif name == 'binomial':
        return make_multi_class_count_contrastive_pairs(n_class=n_class, sim=sim)
    elif name == 'multi-similarity':
        return make_multi_class_count_contrastive_pairs(n_class=n_class, sim=sim, contrastive_margin=1.)
    else:
        raise ValueError("Unknown metric function. Expected 'contrastive', 'triplet', 'binomial' or 'multi-similarity'"
                         " but got {}".format(name))
