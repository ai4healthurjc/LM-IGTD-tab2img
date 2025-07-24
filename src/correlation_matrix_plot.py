import argparse
from pathlib import Path
from utils.dissimilarity import compute_corr_mixed_dataset_2
from utils.preprocessing import identify_feature_type
from utils.loader import load_dataset_with_noise
from utils.plotter import plot_corr_mixed_matrix
import utils.consts as consts
import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    parser.add_argument('--noise_type', default='homogeneous', type=str)

    return parser.parse_args()


parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)
path_dataset, features, y_label = load_dataset_with_noise(args.dataset, args.noise_type)
dataset, categorical, numeric = identify_feature_type(features, categorical_threshold=0.05, impute_missing=False)


df_corr = compute_corr_mixed_dataset_2(features, categorical, numeric)
csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_SAVE_PLOTS, 'Correlation_{}_{}.pdf'.format(args.noise_type, args.dataset)))
plot_corr_mixed_matrix(df_corr, csv_filepath=csv_file_path, figsize=(19, 19), flag_save_figure=True)

