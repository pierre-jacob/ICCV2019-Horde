#!/usr/bin/env python
# coding: utf-8
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

from ..utils.image_processing import preprocess, center_crop
from ..utils.generic_utils import labels2indexes


class DMLGenerator(Sequence):
    """ Keras Sequence for Deep Metric Learning with real-time data augmentation (multi-resolution, random crop,
    horizontal flip, ...) given with 'data_augmentation' param.

    Arguments:
      images: Array of filename or opened images.
      labels: Image labels
      n_class: Number of class per batch.
      image_per_class: Number of image per class per batch.
      pre_process_method: The pre-process method applied on the images before feeding them to the network.
      prediction_dimensions: size of embeddings for all outputs.
      steps_per_epoch: Number of batch computed per epoch.
      im_crop_size: Size that will be feed into the network.
      n_channel: Default 3. Allows to handle Grayscale images such as MNIST.
      shuffle: boolean to randomly sample or not the images per batch.
      bounding_boxes: Should be a tuple with all x_left, x_right, y_top, y_bottom. Standard left/bottom orientation is
      used.
      data_augmentation: list of tuple with (proba, data augmentation object).
    Returns:
      Instance of DMLGenerator which shall be feed into model.fit_generator
    """
    def __init__(self, images: np.ndarray,
                 labels: np.ndarray,
                 n_class: int,
                 image_per_class: int,
                 pre_process_method: str,
                 prediction_dimensions: int or list,
                 steps_per_epoch: int,
                 im_crop_size: tuple = (224, 224),
                 n_channel=3,
                 shuffle: bool = True,
                 bounding_boxes: list = None,
                 data_augmentation: list = None):
        assert np.shape(images)[0] == np.shape(labels)[0], \
            "'images' and 'labels' do not have the same number of images ({} != {}).".format(np.shape(images)[0],
                                                                                             np.shape(labels)[0])
        self.images = images
        self.n_class = n_class
        self.shuffle = shuffle
        self.n_channel = n_channel
        self.original_labels = labels
        self.steps_per_epoch = steps_per_epoch
        self.im_crop_size = list(im_crop_size)
        self.image_per_class = image_per_class
        self.data_augmentation = data_augmentation
        self.pre_process_method = pre_process_method
        self.prediction_dimensions = prediction_dimensions

        self.unique_original_labels = list(set(self.original_labels))
        self.unique_original_labels.sort()

        self.lbl2index, self.index2lbl = labels2indexes(self.unique_original_labels)
        self.labels = np.array([self.lbl2index[o_lbl] for o_lbl in self.original_labels], dtype=np.int32)

        if bounding_boxes is not None:
            self.x1, self.x2, self.y1, self.y2 = bounding_boxes
            self.use_bounding_boxes = True
        else:
            self.use_bounding_boxes = False

        self.ind = self.gen_idx2()

        self.n_images = len(self.ind)  # N pairs in case of training set, or N images in other cases.

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        ind = self.ind[idx]
        im_list = self.images[ind]  # get images

        # Labels for DML: as keras losses enforce that y_pred.shape == y_true.shape, we build full of zeros matrices.
        if isinstance(self.prediction_dimensions, list):
            y_true = []
            for pred_dim in self.prediction_dimensions:
                y_true.append(np.zeros((len(ind), pred_dim), dtype=np.float32))
                y_true[-1][:, 0] = self.labels[ind]
        else:
            y_true = np.zeros((len(ind), self.prediction_dimensions), dtype=np.float32)
            y_true[:, 0] = self.labels[ind]  # get labels and set the into the first column

        batch_images = np.zeros((len(ind),) + tuple(self.im_crop_size) + (self.n_channel,), dtype=np.float32)

        # Generate batch
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

            if self.data_augmentation is not None:
                for proba, data_aug in self.data_augmentation:
                    if np.random.rand() < proba:
                        im = data_aug.compute(im)

            im = center_crop(im, self.im_crop_size)
            # im.show()
            if im.mode == "RBG":
                im = np.array(im, dtype=np.float32)
            elif im.mode == "L":
                im = np.array(im, dtype=np.float32)[..., None]
            batch_images[i, :, :, :] = im

        batch_images = preprocess(batch_images, pre_process_method=self.pre_process_method)

        return batch_images, y_true

    def on_epoch_end(self):
        self.ind = self.gen_idx2()

    def gen_idx2(self):
        """ Pre-computation of indexes to build the batches in __getitem__.

        :return: nd-array: all indexes for each training batch from a given epoch.
        """
        unique_instances = list(set(self.labels))
        unique_instances.sort()

        # Adapt the number of image per class: it avoids to have many time the same image from a given class.
        im_list = []
        to_pop = []
        min_im_per_instance = + np.inf
        for ul in unique_instances:
            ind = np.where(self.labels == ul)[0]
            if len(ind) > 1:
                min_im_per_instance = min(min_im_per_instance, len(ind))
                if self.shuffle:
                    np.random.shuffle(ind)
                im_list.append(ind)
            else:
                to_pop.append(ul)

        for tp in to_pop:
            unique_instances.remove(tp)

        if self.shuffle:
            np.random.shuffle(im_list)

        # The choice is to use at least one time each image as query.
        # Some may occur many times to ensure exclusive batch construction.
        idx_im = np.zeros(len(im_list), dtype=np.int32)
        out_list_of_n_pairs = []

        idx_cls = np.arange(len(im_list))
        if self.shuffle:
            np.random.shuffle(idx_cls)
        i = 0
        for _ in range(self.steps_per_epoch):
            # Get all classes for the current batch:
            idx = idx_cls[np.arange(i, i+self.n_class) % len(im_list)]
            i += self.n_class
            if i >= len(im_list):
                i = 0
                if self.shuffle:
                    np.random.shuffle(idx_cls)

            # Get images from the given classes to build the batch:
            tmp_list = []
            for j in idx:
                l_j = im_list[j]
                for k in range(self.image_per_class):
                    tmp_list.append(l_j[idx_im[j] % len(l_j)])
                    idx_im[j] += 1
                    if idx_im[j] >= len(l_j) and self.shuffle:
                        np.random.shuffle(l_j)

            out_list_of_n_pairs.append(tmp_list)
        out_list_of_n_pairs = np.array(out_list_of_n_pairs, dtype=np.int32)
        return out_list_of_n_pairs

    def train_classes_to_index(self):
        """ Convert labels into indexes.

        :return: two dictionaries for the forward conversion and the backward conversion.
        """
        unique_classes = np.sort(list(set(self.labels)))
        new_labels = np.zeros((len(self.labels),), dtype=np.int32)
        index2class = np.zeros(len(unique_classes), np.int32)

        for i, c in enumerate(unique_classes):
            new_labels[np.where(self.labels == c)] = i
            index2class[i] = c

        # class2index: the new label
        # index2class: the correspondence between the new (index) and the old (value) labels.
        return new_labels, index2class
