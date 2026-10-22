import os
import math
import argparse
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from utils.loader import load_split_dataset
from utils.preprocessing import identify_type_features
from igtd.igtd_functions import table_to_image
import logging
import coloredlogs
import utils.consts as cons

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def get_grid_size(num_features):
    """
    Most square grid with H * W >= d': H = ceil(sqrt(d')), W = ceil(d' / H).
    Pixels without an assigned feature are set to zero.
    """
    num_row = int(math.ceil(math.sqrt(num_features)))
    num_col = int(math.ceil(num_features / num_row))
    return num_row, num_col


def min_max_fit_transform(x_train, x_test):
    """
    Min-max scaling fitted on the training partition and applied to the test partition (test values are clipped to
    the training range). Constant training features are set to 0.
    """
    v_min = x_train.min(axis=0)
    v_range = x_train.max(axis=0) - v_min
    v_range_safe = v_range.replace(0, 1)
    x_train_scaled = (x_train - v_min) / v_range_safe
    x_test_scaled = ((x_test - v_min) / v_range_safe).clip(0, 1)
    x_train_scaled.loc[:, v_range == 0] = 0
    x_test_scaled.loc[:, v_range == 0] = 0
    return x_train_scaled, x_test_scaled


def tab_to_image_train_test(x_train, x_test, path_seed):
    x_test = x_test[x_train.columns]
    x_train_scaled, x_test_scaled = min_max_fit_transform(x_train.astype(float), x_test.astype(float))
    num_row, num_col = get_grid_size(x_train_scaled.shape[1])
    logger.info('Image grid: {} x {} for {} features'.format(num_row, num_col, x_train_scaled.shape[1]))

    # Pixel distance: Euclidean for purely numerical data, Gower for mixed-type data (computed on pixel coordinates)
    distances = args.distances
    if distances == 'auto':
        _, categorical_vars, _ = identify_type_features(x_train, discrete_threshold=10)
        distances = 'Gower' if len(categorical_vars) > 0 else 'Euclidean'
    logger.info('Pixel distance: {}'.format(distances))

    # The feature ranking matrix is computed on the training partition only; the resulting feature-to-pixel
    # assignment is then applied unchanged to the test partition.
    table_to_image(x_train_scaled,
                   [num_row, num_col],
                   args.corr,
                   distances,
                   save_image_size,
                   args.max_step,
                   args.val_step,
                   os.path.join(path_seed, 'train'),
                   args.err,
                   args.interpretability,
                   test_d=x_test_scaled,
                   normDir_test=os.path.join(path_seed, 'test')
                   )


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ionos', type=str)
    parser.add_argument('--noise_type', default='heterogeneous', type=str,
                        help="'preprocessed' (no augmentation), 'homogeneous' (HoNG) or 'heterogeneous' (HeNG)")
    parser.add_argument('--categorical_encoding', default='count', type=str)
    parser.add_argument('--corr', default='mixed', type=str)
    parser.add_argument('--n_jobs', default=50, type=int)
    parser.add_argument('--max_step', default=2000, type=int)
    parser.add_argument('--val_step', default=300, type=int)
    parser.add_argument('--err', default='squared', type=str)
    parser.add_argument('--distances', default='auto', type=str,
                        help="Distance between pixel coordinates: 'auto' (Euclidean for purely numerical data, "
                             "Gower for mixed-type data), Euclidean, Manhattan or Gower")
    parser.add_argument('--interpretability', default=1, type=int)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='LM-IGTD: tabular to image')
args = parse_arguments(parser)
save_image_size = 3  # Size of pictures (in inches) saved during the execution of IGTD algorithm.

n_procs = cpu_count()
n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
logger.info('n_jobs for tab_to_image: {}'.format(n_jobs))

result_dir = os.path.join(cons.PATH_PROJECT_REPORTS_TAB_TO_IMAGE, 'image_{}_{}'.format(args.noise_type, args.dataset))
os.makedirs(name=result_dir, exist_ok=True)

for seed in cons.SEEDS:
    x_train, y_train, x_test, y_test = load_split_dataset(args.dataset, args.noise_type, seed,
                                                          args.categorical_encoding)
    path_seed = os.path.join(result_dir, 'seed_{}'.format(seed))
    os.makedirs(path_seed, exist_ok=True)
    tab_to_image_train_test(x_train, x_test, path_seed)
    # Labels are stored next to the images so that each image can be matched to its class
    pd.DataFrame({'label': list(y_train)}).to_csv(os.path.join(path_seed, 'train', 'labels.csv'), index=False)
    pd.DataFrame({'label': list(y_test)}).to_csv(os.path.join(path_seed, 'test', 'labels.csv'), index=False)
