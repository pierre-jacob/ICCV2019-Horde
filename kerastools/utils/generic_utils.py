#!/usr/bin/env python
# coding: utf-8
import json
import numpy as np
from os.path import join, expanduser


def expanded_join(path, *paths):
    return expanduser(join(path, *paths))


def labels2indexes(lbl):
    lbl2index = {l: i for i, l in enumerate(lbl)}
    index2lbl = {i: l for i, l in enumerate(lbl)}

    return lbl2index, index2lbl


def load_params(filename: str, summary: bool = True):
    with open(filename) as f:
        data = json.load(f)

    if summary:
        summary = "--------------------------------------------------------------------------------"
        before = int(np.floor(len(summary) - len("Script params")) / 2)
        summary += '\n'
        for _ in range(before):
            summary += ' '
        summary += 'Script params'
        summary += '\n--------------------------------------------------------------------------------\n'

        keys = np.sort(list(data.keys()))
        for key in keys:
            summary += '\t' + str(key) + ' : ' + str(data[key]) + '\n'

        summary += "\n--------------------------------------------------------------------------------\n\n"

        print(summary)

    return data
