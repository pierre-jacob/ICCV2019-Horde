#! /usr/bin/env python3
# coding: utf-8
import argparse
import configparser
from tensorflow.keras.optimizers import Adam

from kerastools.databases import Datasets
from kerastools.losses import get_dml_loss
from kerastools.models.abe_models import ABE
from kerastools.models.dml_models import Baseline
from kerastools.callbacks import GlobalMetricCallback
from kerastools.losses.iccv2019_loss import make_abe_loss
from kerastools.metrics.global_metrics import KerasRecallAtK
from kerastools.metrics.batchwise_metrics import get_dml_metric
from kerastools.image_generators.dml_generator import DMLGenerator
from kerastools.image_generators.test_generator import TestGenerator
from kerastools.utils.generic_utils import expanded_join, load_params
from kerastools.models.horde_models import KOrderModel, CascadedKOrder, CascadedABE
from kerastools.utils.image_processing import MultiResolution, RandomCrop, HorizontalFlip


def get_model_and_preprocess(extractor_name, embedding_size, high_order_dims, trainable, cascaded, use_abe, use_horde):
    if use_horde:
        if use_abe:
            return CascadedABE(embedding_size=embedding_size,
                               high_order_dims=high_order_dims,
                               features_reduction=1024,
                               ho_trainable=True,
                               n_head=8)
        elif cascaded:
            return CascadedKOrder(extractor_name=extractor_name,
                                  embedding_sizes=embedding_size,
                                  high_order_dims=high_order_dims,
                                  ho_trainable=trainable)
        else:
            return KOrderModel(extractor_name=extractor_name,
                               embedding_sizes=embedding_size,
                               high_order_dims=high_order_dims,
                               ho_trainable=trainable)
    else:
        if use_abe:
            return ABE(embedding_size=embedding_size)
        else:
            return Baseline(extractor_name=extractor_name, embedding_size=embedding_size)


def get_args():
    parser = argparse.ArgumentParser(description="Script to train DML models with HORDE regularizer (ICCV 2019) "
                                                 "on the 4 datasets used in the original paper.")

    parser.add_argument('--dataset',
                        dest='dataset',
                        required=True,
                        help='Dataset name. Should be in {CUB|CARS|SOP|INSHOP|MNIST}', type=str, default=None)

    parser.add_argument('--feature',
                        dest='feature_extractor',
                        help='Feature extractor name. Should be in {GoogleNet|BNInception}.', type=str, default="GoogleNet")

    parser.add_argument('--loss',
                        dest='loss',
                        help='Loss function used to train the model. Should be in {contrastive|triplet|binomial}.', type=str, default="contrastive")

    parser.add_argument("--model-config",
                        type=str,
                        dest='mdl_cfg',
                        default="config.json",
                        help="Path to the json file which contains all the training parameters.")

    parser.add_argument("--generic-config",
                        type=str,
                        dest='gen_cfg',
                        default="config.ini",
                        help="Path to the file which contains generic information "
                             "(paths to datasets, log folders, ...)")

    parser.add_argument('--embedding',
                        nargs='+',
                        type=int,
                        default=[512],
                        dest='embeddings')

    parser.add_argument('--ho-dim',
                        nargs='+',
                        type=int,
                        default=0,
                        dest='high_order_dims')

    parser.add_argument('--trainable',
                        default=False,
                        action='store_true',
                        dest='trainable')

    parser.add_argument('--cascaded',
                        default=False,
                        action='store_true',
                        dest='cascaded')

    parser.add_argument('--use_abe',
                        default=False,
                        action='store_true',
                        dest='use_abe')

    parser.add_argument('--use_horde',
                        default=False,
                        action='store_true',
                        dest='use_horde')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Load parameters (generic configuration and model specificity) and set paths
    gen_cfg = configparser.ConfigParser()
    gen_cfg.read(expanded_join(args.gen_cfg))

    res_path = gen_cfg['PROJECT_FOLDERS']['RES_PATH']

    mdl_cfg = load_params(expanded_join(args.mdl_cfg), summary=True)

    # Get train/test splits and some information about it:
    dataset = Datasets['retrieval'][args.dataset]()
    train_images, train_labels = dataset.get_training_set()
    test_images, test_labels = dataset.get_testing_set()

    test_indexes_query = dataset.get_queries_idx(db_set='test')
    test_indexes_collection = dataset.get_collection_idx(db_set='test')  # Test metrics are computed only for them
    rank_list = dataset.get_usual_retrieval_rank()  # Usual retrieval ranks

    # Prepare metric on test set:
    test_global_recall_at_K = KerasRecallAtK(ind_queries=test_indexes_query,
                                             ind_collection=test_indexes_collection,
                                             k_list=rank_list,
                                             queries_in_collection=dataset.queries_in_collection)  # R@K metric

    # Preparing loss and metrics:
    dml_loss = get_dml_loss(name=args.loss, n_class=mdl_cfg['n_classes_per_batch'], sim='cosine')
    dml_metric = get_dml_metric(name=args.loss, n_class=mdl_cfg['n_classes_per_batch'], sim='cosine')
    if args.use_abe:
        abe_loss = make_abe_loss(n_class=mdl_cfg['n_classes_per_batch'], n_ensemble=8, lambda_div=0.5)
        if args.use_horde:
            loss = [abe_loss] + [dml_loss] * len(args.high_order_dims)
            loss_weights = [1.] + [1./len(args.high_order_dims)] * len(args.high_order_dims)
        else:
            loss = abe_loss
            loss_weights = [1.]
    else:
        if args.use_horde:
            loss = [dml_loss] * len(args.embeddings)
            loss_weights = [1.] * len(args.embeddings)
        else:
            loss = dml_loss
            loss_weights = [1.]

    args.embeddings = args.embeddings[0] if len(args.embeddings) == 1 else args.embeddings
    # Set model weights name and starting epoch number to resume previous training
    model_name = ""
    if args.cascaded:
        model_name = "Cascaded"
    else:
        model_name = "Full"

    if not args.trainable:
        model_name += "_RM"

    model_weights_filename = "{}_{}_{}_weights".format(args.feature_extractor,
                                                       model_name,
                                                       dataset.name)
    # Build model:
    model, pre_process_method = get_model_and_preprocess(extractor_name=args.feature_extractor,
                                                         embedding_size=args.embeddings,
                                                         high_order_dims=args.high_order_dims,
                                                         trainable=args.trainable,
                                                         cascaded=args.cascaded,
                                                         use_abe=args.use_abe,
                                                         use_horde=args.use_horde)
    if isinstance(model.output_shape, tuple):
        output_dim = model.output_shape[-1]
    elif isinstance(model.output_shape, list):
        output_dim = [out_shape[-1] for out_shape in model.output_shape]
    else:
        raise ValueError("Expected output. Expected a 2d-tuple (=shape) or a list (of 2d-shapes)."
                         "Got {}.".format(model.output_shape))

    # Load weights if necessary and sum up model:
    if mdl_cfg['resume_training'] is not None:
        model.load_weights(expanded_join(res_path, model_weights_filename +
                                         "_" + mdl_cfg['resume_training'] + ".h5"))

    # Build data augmentation functions:
    data_aug = [
        (mdl_cfg['proba_multi_res'], MultiResolution(crop_size=mdl_cfg['train_im_size'],
                                                     max_ratio=mdl_cfg['multi_res_max_ratio'],
                                                     min_ratio=mdl_cfg['multi_res_min_ratio'],
                                                     prob_keep_aspect_ratio=mdl_cfg['prob_keep_ratio'])),
        (mdl_cfg['proba_random_crop'], RandomCrop(crop_size=mdl_cfg['train_im_size'])),
        (mdl_cfg['proba_horizontal_flip'], HorizontalFlip())
    ]

    # Build data generators for training and testing sets:
    train_data_gen = DMLGenerator(images=train_images,
                                  labels=train_labels,
                                  n_class=mdl_cfg['n_classes_per_batch'],
                                  image_per_class=mdl_cfg['images_per_class'],
                                  pre_process_method=pre_process_method,
                                  prediction_dimensions=output_dim,
                                  im_crop_size=mdl_cfg['train_im_size'],
                                  steps_per_epoch=mdl_cfg['steps_per_epoch'],
                                  shuffle=True,
                                  data_augmentation=data_aug)

    test_data_gen = TestGenerator(images=test_images,
                                  labels=test_labels,
                                  pre_process_method=pre_process_method,
                                  im_crop_size=mdl_cfg['test_im_size'],
                                  batch_size=mdl_cfg['test_batch_size'],
                                  ratio=1.144,
                                  keep_aspect_ratio=False)

    metrics_cb = GlobalMetricCallback(data_gen=test_data_gen,
                                      frequency=mdl_cfg['compute_scores_freq'],
                                      list_of_metrics=[test_global_recall_at_K],
                                      max_queue_size=mdl_cfg['max_queue_size'],
                                      workers=mdl_cfg['workers'],
                                      use_multiprocessing=False,
                                      save_best_representations=True,
                                      save_as_hdf=args.use_horde,
                                      path_to_save='logs',
                                      generic_filename=model_weights_filename[:-len('_weights')])

    model.compile(optimizer=Adam(lr=mdl_cfg['train_lr']),
                  loss=loss,
                  loss_weights=loss_weights)

    model.summary()
    model.fit_generator(generator=train_data_gen,
                        epochs=mdl_cfg['train_epoch'],
                        callbacks=[metrics_cb],
                        validation_data=None,
                        validation_steps=None,
                        shuffle=True,
                        initial_epoch=0,
                        workers=mdl_cfg['workers'],
                        max_queue_size=mdl_cfg['max_queue_size'],
                        use_multiprocessing=False)

    print("job's done.")
