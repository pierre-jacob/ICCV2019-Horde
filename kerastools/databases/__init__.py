#!/usr/bin/env python
# coding: utf-8
from .cars196 import Cars196Ret
from .cub_200_2011 import Cub200Ret
from .stanford_online_products import StanfordOnlineProducts
from .inshop import InShop
from .mnist import MnistRet

Datasets = {'retrieval': {'CUB': Cub200Ret,
                          'CARS': Cars196Ret,
                          'INSHOP': InShop,
                          'SOP': StanfordOnlineProducts,
                          'MNIST': MnistRet}}
