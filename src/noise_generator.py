import time
import argparse
import logging
import coloredlogs
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import utils.consts as consts
from utils.loader import load_split_dataset, get_split_dataset_path
from utils.preprocessing import preprocess_for_clustering
from utils.noise_generator import NoiseFeatureGenerator
from utils.plotter import plot_TSNE, plot_UMAP

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def spectral_clustering_gower(dist_matrix, n_clusters):
    """
    Spectral clustering on a Gower distance matrix. The affinity is a Gaussian kernel,
    S_ij = exp(-d_G(z_i, z_j)^2 / (2 h^2)), with the bandwidth h set to the median pairwise distance.
    """
    off_diagonal = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    h = np.median(off_diagonal)
    if h <= 0:
        h = np.mean(off_diagonal) if np.mean(off_diagonal) > 0 else 1.0
    affinity = np.exp(-dist_matrix ** 2 / (2 * h ** 2))
    model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    return model.fit_predict(affinity)


def silhouette_gower(dist_matrix, labels):
    if len(set(labels)) < 2:
        return -1.0
    return silhouette_score(dist_matrix, labels, metric='precomputed')


def select_n_clusters(dist_matrix, k_values=range(2, 11)):
    """K_clust in {2, ..., 10} maximizing the Silhouette Coefficient on the training partition."""
    best = (None, -np.inf, None)
    for k in k_values:
        labels = spectral_clustering_gower(dist_matrix, k)
        score = silhouette_gower(dist_matrix, labels)
        if score > best[1]:
            best = (k, score, labels)
    return best


def evaluate_candidate(x_train, noise_type, n_new_vars, n_clusters, candidate_seed):
    """Generates candidate Z_t, re-clusters it with the fixed K_clust and returns its Silhouette."""
    generator = NoiseFeatureGenerator(noise_type, n_new_vars, seed=candidate_seed).fit(x_train)
    z_train = generator.transform(x_train, partition_seed=0)
    dist_matrix, _ = preprocess_for_clustering(z_train)
    labels = spectral_clustering_gower(dist_matrix, n_clusters)
    return candidate_seed, silhouette_gower(dist_matrix, labels), generator


def generate_n_noisy_features(x_train, x_test, noise_type, n_new_vars, n_boots, n_jobs, save_name):
    """
    Selects the augmentation on the training partition (N candidates, spectral clustering with fixed K_clust and
    Silhouette-based selection) and applies the selected configuration, unchanged, to the test partition.
    """
    x_train = x_train.reset_index(drop=True)
    x_test = x_test[x_train.columns].reset_index(drop=True)

    dist_train, _ = preprocess_for_clustering(x_train)
    n_clusters, baseline_score, baseline_labels = select_n_clusters(dist_train)
    logger.info('K_clust = {} (Silhouette on the original training data = {:.3f})'.format(n_clusters, baseline_score))

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_candidate)(x_train, noise_type, n_new_vars, n_clusters, t) for t in range(n_boots))
    best_seed, best_score, generator = max(results, key=lambda r: r[1])
    logger.info('Selected candidate {} (Silhouette = {:.3f}, {} noisy features)'.format(
        best_seed, best_score, len(generator.config)))

    df_train_noisy = generator.transform(x_train, partition_seed=0)
    df_test_noisy = generator.transform(x_test, partition_seed=1)

    # t-SNE / UMAP projections of the original and the selected augmented training data
    dist_best, _ = preprocess_for_clustering(df_train_noisy)
    for dist_matrix, score, tag in [(dist_train, baseline_score, 'baseline'), (dist_best, best_score, 'noise')]:
        name = '{}_{}_{}'.format(save_name, tag, np.round(score, 3))
        plot_TSNE(dist_matrix, name, baseline_labels, consts.PATH_PROJECT_REPORTS_NOISY_FIGURES)
        plot_UMAP(dist_matrix, 20, name, baseline_labels, consts.PATH_PROJECT_REPORTS_NOISY_FIGURES)

    return df_train_noisy, df_test_noisy


def save_split_results(df_concat, y_label, noise_type, dataset, seed, partition):
    df_concat_with_labels = df_concat.reset_index(drop=True)
    df_concat_with_labels['label'] = list(y_label)
    csv_file_path = str(get_split_dataset_path(dataset, noise_type, seed, partition))
    df_concat_with_labels.to_csv(csv_file_path, index=False)
    logger.info('Saved {} ({} samples, {} features)'.format(csv_file_path, df_concat_with_labels.shape[0],
                                                            df_concat_with_labels.shape[1] - 1))


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ionos', type=str)
    parser.add_argument('--categorical_encoding', default='count', type=str)
    parser.add_argument('--n_jobs', default=15, type=int)
    parser.add_argument('--n_boots', default=100, type=int)  # N = 100 candidate augmentations
    parser.add_argument('--n_new_vars', default=3, type=int)  # M = 3 noisy variants per feature
    parser.add_argument('--noise_type', default='heterogeneous', type=str)  # homogeneous (HoNG) / heterogeneous (HeNG)
    return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser(description='noise generator')
    args = parse_arguments(parser)

    for seed in consts.SEEDS:
        # The augmentation is selected on the training partition only; the selected configuration is applied to the
        # test partition, whose noisy variants are generated once.
        x_train, y_train, x_test, y_test = load_split_dataset(args.dataset, 'preprocessed', seed,
                                                              args.categorical_encoding)

        df_train_noisy, df_test_noisy = generate_n_noisy_features(
            x_train, x_test, args.noise_type, args.n_new_vars, args.n_boots, args.n_jobs,
            save_name='_'.join([args.dataset, args.noise_type, args.categorical_encoding, 'seed', str(seed)]))

        save_split_results(df_train_noisy, y_train, args.noise_type, args.dataset, seed, 'train')
        save_split_results(df_test_noisy, y_test, args.noise_type, args.dataset, seed, 'test')

    logger.info('Execution time: {:.1f} s'.format(time.time() - start_time))
