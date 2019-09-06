#!/usr/bin/env python
# coding: utf-8
import numpy as np
from copy import deepcopy
from abc import abstractmethod, ABC
from PIL import Image, ImageOps, ImageColor
from keras_applications.imagenet_utils import preprocess_input

dic_mine_to_keras = {'imagenet_centered': 'caffe',
                     'imagenet_reduced': 'torch',
                     'reduced': 'tf'}


def preprocess(x, pre_process_method):
    if pre_process_method not in dic_mine_to_keras.keys():
        raise ValueError("mode {} doesn't supported. Expected values: {}".format(pre_process_method, dic_mine_to_keras.keys()))
    if isinstance(x, np.ndarray):
        t = deepcopy(x)
    else:
        t = x
    return preprocess_input(x=t, mode=dic_mine_to_keras[pre_process_method], data_format='channels_last')


class DataAugmentation(ABC):
    """ Abstract data augmentation class.

    Arguments:

    Returns:

    """

    @abstractmethod
    def compute(self, x: Image):
        """ This function apply the data augmentation on the given Image object.
        """
        raise NotImplementedError("The method 'compute' has not been implemented.")


class MultiResolution(DataAugmentation):
    """ Multi-resolution as a data augmentation.

    Arguments:
      crop_size: output crop size
      max_ratio: max ratio to compute the resolution according to the crop size
      min_ratio: min ratio to compute the resolution according to the crop size
      prob_keep_aspect_ratio: probability to keep the aspect ratio of the input image
      interpolation_method: Interpolation method for image resizing.
    Returns:
      Instance of DataAugmentation to use in ImageGenerator.
    """
    def __init__(self, crop_size: tuple,
                 max_ratio: float,
                 min_ratio: float,
                 prob_keep_aspect_ratio=0.5,
                 interpolation_method=Image.LANCZOS):
        self.crop_size = crop_size
        self.interpolation_method = interpolation_method
        self.prob_keep_aspect_ratio = prob_keep_aspect_ratio
        self.min_crop_size = (int(crop_size[0] * min_ratio), int(crop_size[1] * min_ratio))
        self.max_crop_size = (int(crop_size[0] * max_ratio), int(crop_size[1] * max_ratio))

    def compute(self, x: Image):
        if np.random.rand() < self.prob_keep_aspect_ratio:
            w, h = x.size
            r = np.random.rand()
            if h > w:
                h_new = self.min_crop_size[0] + r * (self.max_crop_size[0] - self.min_crop_size[0])
                w_new = w / h * h_new
            else:
                w_new = self.min_crop_size[1] + r * (self.max_crop_size[1] - self.min_crop_size[1])
                h_new = h / w * w_new
        else:
            r = np.random.rand()
            h_new = self.min_crop_size[0] + r * (self.max_crop_size[0] - self.min_crop_size[0])
            w_new = self.min_crop_size[1] + r * (self.max_crop_size[1] - self.min_crop_size[1])

        h_new = int(np.ceil(h_new))
        w_new = int(np.ceil(w_new))

        x = x.resize((w_new, h_new), self.interpolation_method)
        return x


class RandomCrop(DataAugmentation):
    """ Random crop as a data augmentation.

    Arguments:
      crop_size: output crop size
      bg_color: color of the background if needed
    Returns:
      Instance of DataAugmentation to use in ImageGenerator.
    """
    def __init__(self, crop_size=(224, 224), bg_color=ImageColor.colormap['black']):
        self.bg_color = bg_color
        self.crop_size = crop_size

    def compute(self, x: Image):
        w, h = x.size
        if w == self.crop_size[1] and h == self.crop_size[0]:
            return x

        h_off_range = h - self.crop_size[0]
        w_off_range = w - self.crop_size[1]

        if h_off_range > 0:
            h_off = np.random.randint(0, h_off_range, 1, dtype=np.int32)[0]
            h_m = self.crop_size[0]
        else:
            h_off = 0
            h_m = h

        if w_off_range > 0:
            w_off = np.random.randint(0, w_off_range, 1, dtype=np.int32)[0]
            w_m = self.crop_size[1]
        else:
            w_off = 0
            w_m = w

        x = x.crop((w_off, h_off, w_off + w_m, h_off + h_m))
        new_x = Image.new("RGB", (self.crop_size[1], self.crop_size[0]), color=self.bg_color)
        new_x.paste(x, ((self.crop_size[1] - w_m) // 2,
                        (self.crop_size[0] - h_m) // 2))

        return new_x


class HorizontalFlip(DataAugmentation):
    """ Vertical flip as a data augmentation.

    Arguments:

    Returns:
      Instance of DataAugmentation to use in ImageGenerator.
    """
    def compute(self, x: Image):
        return ImageOps.mirror(x)


def center_crop(x, crop_size=(224, 224), bg_color=ImageColor.colormap['black']):
    w, h = x.size
    if w == crop_size[1] and h == crop_size[0]:
        return x

    h_off_range = h - crop_size[0]
    w_off_range = w - crop_size[1]

    if h_off_range > 0:
        h_off = int((h - crop_size[0]) / 2)
        h_m = crop_size[0]
    else:
        h_off = 0
        h_m = h

    if w_off_range > 0:
        w_off = int((w - crop_size[1]) / 2)
        w_m = crop_size[1]
    else:
        w_off = 0
        w_m = w

    x = x.crop((w_off, h_off, w_off + w_m, h_off + h_m))
    new_x = Image.new("RGB", (crop_size[1], crop_size[0]), color=bg_color)
    new_x.paste(x, ((crop_size[1] - w_m) // 2,
                    (crop_size[0] - h_m) // 2))

    return new_x
