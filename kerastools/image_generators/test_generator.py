#!/usr/bin/env python
# coding: utf-8
import warnings
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

from ..utils.generic_utils import labels2indexes
from ..utils.image_processing import RandomCrop, HorizontalFlip, preprocess, center_crop


class TestGenerator(Sequence):
    """ Keras sequence for online processing of images.

    Arguments:
      images: Array of filename or opened images.
      labels: Image labels
      pre_process_method: The pre-process method applied on the images before feeding them to the network.
      im_crop_size: Size that will be feed into the network.
      batch_size:
      n_channel: Default 3. Allows to handle grayscale images such as MNIST by using 1.
      bounding_boxes: Should be a tuple with all x_low, x_high, y_low, y_high
      multi_crop_number: Number of multi-crop to be averaged. Default is without multi-crop.
      interpolation_method: The interpolation method to change the image size.
      ratio: 'im_crop_size' ratio gives the image size from which we use a center crop.
      keep_aspect_ratio:
      use_multi_crop_flip: Perform horizontal flip in case of multi-crop.
    Returns:
      Instance of TestDataGenerator which shall be feed into model.predict_generator or model.evaluate_generator
    """
    def __init__(self, images: np.ndarray,
                 labels: np.ndarray,
                 pre_process_method: str,
                 im_crop_size=(224, 224),
                 batch_size=32,
                 n_channel=3,
                 bounding_boxes=None,
                 multi_crop_number: int=0,
                 interpolation_method=Image.LANCZOS,
                 ratio=1.15,
                 keep_aspect_ratio=True,
                 use_multi_crop_flip=False):
        assert np.shape(images)[0] == np.shape(labels)[0], \
            "'images' and 'labels' do not have the same number of images ({} != {}).".format(np.shape(images)[0],
                                                                                             np.shape(labels)[0])
        self.ratio = ratio
        self.images = images
        self.n_channel = n_channel
        self.batch_size = batch_size
        self.original_labels = labels
        self.im_crop_size = list(im_crop_size)
        self.keep_aspect_ratio = keep_aspect_ratio
        self.pre_process_method = pre_process_method
        self.use_multi_crop_flip = use_multi_crop_flip
        self.interpolation_method = interpolation_method

        if multi_crop_number > 1:
            self.n_crop = multi_crop_number
            if multi_crop_number == 2 and use_multi_crop_flip:
                self.random_crop_generator = HorizontalFlip()
            else:
                self.random_crop_generator = RandomCrop(crop_size=self.im_crop_size)
            warnings.warn("Real batch size will be %d "
                          "(number of crop by given batch size)." % (multi_crop_number * batch_size))
        else:
            self.n_crop = 1

        self.unique_original_labels = list(set(self.original_labels))
        self.unique_original_labels.sort()

        self.lbl2index, self.index2lbl = labels2indexes(self.unique_original_labels)
        self.labels = np.array([self.lbl2index[o_lbl] for o_lbl in self.original_labels], dtype=np.int32)

        if bounding_boxes is not None:
            self.x1, self.x2, self.y1, self.y2 = bounding_boxes
            self.use_bounding_boxes = True
        else:
            self.use_bounding_boxes = False

        self.ind = np.arange(len(images), dtype=np.int32)
        self.n_images = len(self.ind)

    def __len__(self):
        return int(np.ceil(self.n_images / float(self.batch_size)))

    def __getitem__(self, idx):
        b = idx * self.batch_size
        N = min(self.batch_size, self.n_images - b)

        ind = self.ind[b:b + N]
        ind = ind.flatten()
        im_list = self.images[ind]

        batch_images = np.zeros((self.n_crop * len(ind),) + tuple(self.im_crop_size) + (self.n_channel,), dtype=np.float32)

        # Generate data
        for i, image in enumerate(im_list):
            if isinstance(image, str):
                im = Image.open(image)
                if self.n_channel == 3:
                    im = im.convert("RGB")
                elif self.n_channel == 1:
                    im = im.convert("L")
                else:
                    raise ValueError("Unexpected channel number. Expected '1' or '3'"
                                     "but got {}".format(self.n_channel))

            elif isinstance(image, np.ndarray):
                if image.shape[-1] == 3:
                    im = Image.fromarray(image, mode="RGB")
                elif image.shape[-1] == 1:
                    im = Image.fromarray(image[..., 0], mode="L")
                else:
                    raise ValueError("Unexpected image size. Expected (h, w, 1) or (h, w, 3.) "
                                     "but got {}".format(image.shape))
            else:
                raise ValueError('Image format not understood.'
                                 'Expected ndarray or str, but got'.format(type(image)))

            if self.use_bounding_boxes:
                im = im.crop((self.x1[ind[i]], self.y1[ind[i]], self.x2[ind[i]], self.y2[ind[i]]))

            if self.keep_aspect_ratio:
                w, h = im.size
                r = max([self.im_crop_size[0] / h, self.im_crop_size[1] / w])
                h = int(np.ceil(h * r * self.ratio))
                w = int(np.ceil(w * r * self.ratio))
                im = im.resize((w, h), self.interpolation_method)
            else:
                target_size = list(self.im_crop_size)
                target_size[0] = int(np.ceil(target_size[0] * self.ratio))
                target_size[1] = int(np.ceil(target_size[1] * self.ratio))
                im = im.resize((target_size[1], target_size[0]), self.interpolation_method)

            if self.n_crop > 1:
                for j in range(self.n_crop):
                    crop = self.random_crop_generator.compute(im)
                    crop = center_crop(crop, self.im_crop_size)
                    if im.mode == "RBG":
                        crop = np.array(crop, dtype=np.float32)
                    elif im.mode == "L":
                        crop = np.array(crop, dtype=np.float32)[..., None]
                    batch_images[i, :, :, :] = crop
            else:
                im = center_crop(im, self.im_crop_size)
                if im.mode == "RBG":
                    im = np.array(im, dtype=np.float32)
                elif im.mode == "L":
                    im = np.array(im, dtype=np.float32)[..., None]
                batch_images[i, :, :, :] = im

        batch_images = preprocess(batch_images, pre_process_method=self.pre_process_method)

        return batch_images
