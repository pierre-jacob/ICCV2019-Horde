#!/usr/bin/env python
# coding: utf-8
import h5py
import numpy as np
import configparser
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

from ..utils.generic_utils import expanded_join


class GlobalMetricCallback(Callback):
    """ Keras Callback to compute global metrics (supposed to be 'on_epoch_end').

    Arguments:
      data_gen: Data generator which pre-processes images and labels.
      frequency: Epoch frequency to compute the metric.
      list_of_metrics: A list which contains all global metrics instances. See metrics.global_metrics for available one.
      save_best_representations: Save or not the predictions produced by the network.
      path_to_save: Path where weights and predictions will be saved. 'logs' default parse the config file.
      generic_filename: Filename which is extended to save weights and predictions.
      max_queue_size: Batch queue size.
      workers: Number of workers to pre-process the batches.
      use_multiprocessing: Python multi-processes.
      monitored_output: which output name to monitor. None monitors all and saves data when one metric is improved.
      save_as_hdf: In the case of multi-output model, save as hdf5 is set to true.
      cbk_model: None by default. If a model is set, this model is saved instead of the one in the callback. This
      allows to handle multi-gpu models and testing parts of a full models.
      verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
    Returns:
      A callback to use in Keras.
    """
    def __init__(self,
                 data_gen,
                 frequency=1,
                 list_of_metrics=[],
                 save_best_representations=False,
                 path_to_save='logs',
                 generic_filename=None,
                 max_queue_size=10,
                 workers=4,
                 use_multiprocessing=True,
                 monitored_output=None,
                 save_as_hdf=False,
                 last_prediction_layer=None,
                 cbk_model=None,
                 verbose=1):
        super().__init__()
        self.data_gen = data_gen
        self.workers = workers
        self.verbose = verbose
        self.frequency = frequency
        self.cbk_model = cbk_model
        self.save_as_hdf = save_as_hdf
        self.max_queue_size = max_queue_size
        self.list_of_metrics = list_of_metrics
        self.generic_filename = generic_filename
        self.monitored_output = monitored_output
        self.use_multiprocessing = use_multiprocessing
        self.last_prediction_layer = last_prediction_layer
        self.save_best_representations = save_best_representations  # default is the first metric (generally, top1)

        if path_to_save == 'logs':
            config = configparser.ConfigParser()
            config.read(expanded_join('config.ini'))

            self.path_to_save = config['PROJECT_FOLDERS']['LOG_PATH']
        else:
            self.path_to_save = path_to_save

        self.best_score = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency == 0:
            print('Computing predictions...')
            # Select the correct model for evaluation:
            if self.last_prediction_layer is not None:
                mdl = Model(self.model.input, self.model.get_layer(name=self.last_prediction_layer).output)
            elif self.cbk_model is not None:
                mdl = self.cbk_model
            else:
                mdl = self.model
            predictions = mdl.predict_generator(generator=self.data_gen,
                                                max_queue_size=self.max_queue_size,
                                                workers=self.workers,
                                                use_multiprocessing=self.use_multiprocessing,
                                                verbose=self.verbose)
            tried_to_save = False
            preds_dict = {}
            print('Computing metrics...')
            if not isinstance(predictions, list):  # Handle multiple outputs models
                predictions = [predictions]
            message = ""
            output_names = mdl.output_names  # Not beautiful, but allow separation.
            for i in range(len(output_names)):
                message += 'Results for {}:\n'.format(output_names[i])
                for metric in self.list_of_metrics:
                    if hasattr(self.data_gen, 'n_crop'):
                        if self.data_gen.n_crop > 1:
                            pred_i = np.reshape(predictions[i], (self.data_gen.n_images, self.data_gen.n_crop, -1))
                            pred_i = np.sum(pred_i, axis=1)
                            pred_i /= np.linalg.norm(pred_i, axis=-1, keepdims=True)
                        else:
                            pred_i = predictions[i]
                    else:
                        pred_i = predictions[i]

                    preds_dict[output_names[i]] = pred_i
                    all_res = metric.compute_metric(pred_i, self.data_gen.labels)
                    if self.monitored_output == output_names[i] or self.monitored_output is None:
                        if self.best_score < all_res[0][1] and not tried_to_save and self.save_best_representations:
                            metric_name = metric.name + str(int(all_res[0][0]))
                            old = self.best_score
                            new = all_res[0][1]
                            print("Saving model and representations: {metric_name} "
                                  "has been increased ({old:3.2f} --> {new:3.2f}).".format(metric_name=metric_name,
                                                                                           old=old,
                                                                                           new=new))
                            tried_to_save = True
                            self.best_score = new

                            if not self.save_as_hdf:
                                np.save(expanded_join(self.path_to_save,
                                                      self.generic_filename +
                                                      "_preds_{metric}_{score:3.2f}_epoch"
                                                      "_{epoch:04d}.npy".format(epoch=epoch,
                                                                                metric=metric_name,
                                                                                score=self.best_score)), predictions)

                    for param, res in all_res:
                        message += output_names[i] + ' : ' + metric.name + str(int(param)) + ': ' + str(res) + '\n'
                    message += '\n'  # Separation between metrics

            if tried_to_save:
                if self.save_as_hdf:
                    f = h5py.File(expanded_join(self.path_to_save,
                                                self.generic_filename +
                                                "_preds_{epoch:04d}.npy".format(epoch=epoch)), 'a')

                    for k, v in preds_dict.items():
                        if k in f:
                            del f[k]  # allows overwriting.
                            f.create_dataset(name=k, shape=np.shape(v), dtype=np.float32, data=v)

                else:
                    mdl.save_weights(expanded_join(self.path_to_save,
                                                   self.generic_filename +
                                                   '_best_weights_epoch_{epoch:04d}.h5'.format(epoch=epoch)))
            print(message)
