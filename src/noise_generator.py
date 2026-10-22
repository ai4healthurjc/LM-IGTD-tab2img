import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from operator import itemgetter
import coloredlogs
import logging
import time
from joblib import Parallel, delayed
from utils.loader import load_split_dataset, get_split_dataset_path
from utils.preprocessing import preprocess_for_clustering, preprocess_for_fs
import utils.consts as consts
from utils.noise_generator import NoiseFeatureGenerator
from typing import Dict
import utils.consts as cons
from utils.plotter import plot_cvi, plot_TSNE, plot_UMAP
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from validclust.indices import dunn, cop
import hdbscan
import random
from features.ensemble_fs import perform_ensemble_fs
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def compute_cvi(x_data, name):
    dist_matrix, data_scaled = preprocess_for_clustering(x_data)
    ss = []
    ch = []
    dun = []
    db = []
    cop_final = []
    for e in np.arange(2, 11, 1):  # K_clust in {2, ..., 10}
        ss_mean = []
        ch_mean = []
        dun_mean = []
        db_mean = []
        cop_mean = []
        for i in range(5):
            spectral_model = SpectralClustering(n_jobs=-1, n_clusters=e, eigen_solver='lobpcg',
                                                n_neighbors=int(0.1 * len(x_data)),
                                                affinity='precomputed_nearest_neighbors', random_state=i)
            labels_rbf = spectral_model.fit_predict(dist_matrix)
            ss_mean.append(silhouette_score(dist_matrix, labels_rbf, metric="precomputed"))
            ch_mean.append(calinski_harabasz_score(dist_matrix, labels_rbf))
            db_mean.append(davies_bouldin_score(dist_matrix, labels_rbf))
            dun_mean.append(dunn(np.asarray(dist_matrix), labels_rbf))
            cop_mean.append(cop(np.asarray(data_scaled), np.asarray(dist_matrix), labels_rbf))

        ss.append(np.mean(ss_mean))
        ch.append(np.mean(ch_mean))
        dun.append(np.mean(dun_mean))
        db.append(np.mean(db_mean))
        cop_final.append(np.mean(cop_mean))

    plot_cvi([ss, ch, dun, db, cop_final],
             ['Shilouette score', 'Calinski-Harabasz', 'Dunn index', 'davies_bouldin', 'COP'],
             name, cons.PATH_PROJECT_REPORTS_NOISY_FIGURES)

    return [ss, ch, dun, db, cop_final]


def generate_n_noisy_features(df_data,
                              labels,
                              n_new_features,
                              clustering_method,
                              noise_type,
                              n_boots,
                              min_n=0.1,
                              max_n=0.6,
                              delta_n=0.05,
                              n_jobs=1,
                              save_name='',
                              dataset2=None
                              ):

    """
    n_new_features: int between [1, 4]
    """
    n_samples, n_features = df_data.shape
    cvi_list = compute_cvi(df_data, save_name)
    pos_max = np.argmax(np.asarray(cvi_list[0]))
    v_predicted_clustering, baseline_score, detected_clusters = compute_clustering(df_data, clustering_method,
                                                                                   n_clusters=pos_max + 2,
                                                                                   baseline_score=np.max(
                                                                                       np.asarray(cvi_list[0])),
                                                                                   plot=True,
                                                                                   save_name=save_name)
    print('Baseline score: ', baseline_score)
    list_random_noise_power = []

    for boot in range(n_boots):
        np.random.seed(boot)
        list_random_noise_power.append(
            np.random.choice(np.arange(min_n, max_n, delta_n), n_new_features * n_features, replace=True)
        )

    list_dict_dfs = Parallel(n_jobs=n_jobs)(
        delayed(get_bootstrap_sample)(df_data, v_noise, labels,n_new_features, noise_type, seed_value, dataset2) for
        seed_value, v_noise in enumerate(list_random_noise_power)
    )
    list_dict_clustering_results = Parallel(n_jobs=n_jobs)(
        delayed(train_fs_method_by_bootstrap)(dict_df, clustering_method, detected_clusters, baseline_score,
                                              save_name,v_predicted_clustering)
        for dict_df in list_dict_dfs)

    df_best_concat, df_best_concat2,dict_best = get_best_subset(list_dict_clustering_results)

    dist_matrix, data_scaled = preprocess_for_clustering(df_best_concat)
    represent_UMAP(dist_matrix, dict_best['score'], baseline_score, v_predicted_clustering, save_name)
    represent_TSNE(dist_matrix, dict_best['score'], baseline_score, v_predicted_clustering, save_name)

    return df_best_concat, df_best_concat2


def get_best_subset(list_dict_clustering_results):
    df = pd.DataFrame()
    df['id_boot'] = list(map(itemgetter('id_boot'), list_dict_clustering_results))
    df['score'] = list(map(itemgetter('score'), list_dict_clustering_results))

    best_sil_score = df.iloc[df['score'].idxmax()]
    best_id_boot = df.iloc[df['score'].idxmax()]['id_boot'].item()
    dict_best_concat = list(filter(lambda dict_cluster_results: dict_cluster_results['id_boot'] == best_id_boot,
                                   list_dict_clustering_results))[0]

    print('best boot: {}'.format(best_id_boot))
    print('best boot: {}'.format(best_sil_score))

    return dict_best_concat['df_concat'],dict_best_concat['df_concat2'],dict_best_concat
def create_noisy_dataset(df_data,noise_type,labels,noise_generator,n_new_vars,random_state):
    df_data_noisy = noise_generator.transform(df_data.copy())
    if noise_type == 'homogeneous':
        df_concat = pd.concat([df_data.copy(), df_data_noisy], axis=1)
    elif noise_type == 'heterogeneous':
        df_concat= select_noisy_features(df_data, df_data_noisy,labels, n_new_vars, random_state)
    elif noise_type == 'heterogeneous_squared':
        df_concat= select_noisy_features(df_data, df_data_noisy,labels, n_new_vars, random_state,squared=True)
    else:
        raise ValueError("Incorrect noise type: choose between homogeneous and heterogeneous")
    return df_concat
def create_two_noisy_dataset(df_data,df_data2,noise_type,labels,noise_generator,n_new_vars,random_state):
    # df_data: training partition, df_data2: test partition. The noise model is fitted on the training partition.
    noise_generator.fit(df_data)
    df_data_noisy = noise_generator.transform(df_data.copy())
    df_data_noisy2 = noise_generator.transform(df_data2.copy())
    df_concat2 = pd.concat([df_data2.copy(), df_data_noisy2], axis=1)
    if noise_type == 'homogeneous':
        df_concat = pd.concat([df_data.copy(), df_data_noisy], axis=1)
        df_concat2 = df_concat2[df_concat.columns]
    elif noise_type == 'heterogeneous_squared':
        df_concat= select_noisy_features(df_data, df_data_noisy,labels, n_new_vars, random_state,squared=True)
        df_concat2=df_concat2[df_concat.columns]
    elif noise_type == 'heterogeneous':
        df_concat= select_noisy_features(df_data, df_data_noisy,labels, n_new_vars, random_state)
        df_concat2=df_concat2[df_concat.columns]
    else:
        raise ValueError("Incorrect noise type: choose between homogeneous, heterogeneous or heterogeneous_squared")
    return df_concat,df_concat2

def get_bootstrap_sample(df_data: pd.DataFrame,
                         v_noise: np.array,
                         labels: np.array,
                         n_new_vars: int,
                         noise_type: str,
                         random_state: int,
                         df_data2: pd.DataFrame
                         ) -> Dict:
    nfg = NoiseFeatureGenerator(v_noise, df_data.shape[1], n_new_vars, random_state)

    if df_data2 is None:
        df_concat = create_noisy_dataset(df_data, noise_type,labels, nfg, n_new_vars, random_state)
        df_concat2 = pd.DataFrame()
    else:
        df_concat, df_concat2 = create_two_noisy_dataset(df_data,df_data2, noise_type,labels, nfg, n_new_vars, random_state)

    dict_result = {
        'id_boot': random_state,
        'df_concat': df_concat,
        'df_concat2': df_concat2
    }

    return dict_result

def sum_score(df_score):
    val=['score0', 'score1', 'score2', 'score3', 'score4']
    df_score['std']=df_score[val].astype(float).T.std()
    df_score['score_sum']=df_score[val].astype(float).T.mean()
    df_score.sort_values(by=['score_sum'], ascending=False, inplace=True)
    return df_score

def squared_noise(df_data,n_new_vars):
    create_n_var = []
    minimal_n_var_aux = np.round(np.sqrt(df_data.shape[1])+0.4, 0)
    minimal_n_var = minimal_n_var_aux * minimal_n_var_aux - df_data.shape[1]
    create_n_var.append(minimal_n_var)

    while np.max(create_n_var) < n_new_vars * df_data.shape[1]:
        minimal_n_var_aux += 1
        minimal_n_var_ = minimal_n_var_aux * minimal_n_var_aux - df_data.shape[1]
        if minimal_n_var_ >= df_data.shape[1]:
            create_n_var.append(minimal_n_var_)
    create_n_var.remove(minimal_n_var)
    return create_n_var
def select_noisy_features(df_data, df_data_noisy,labels, n_new_vars, random_state,method= 'random',squared=False):
    if squared:
        create_n_var=squared_noise(df_data,n_new_vars)
    else:
        create_n_var=[]
        for i in np.arange(df_data_noisy.shape[1]):
            if i >= df_data.shape[1]:
                create_n_var.append(i)

    random.seed(random_state)
    n_select_var_index = random.randint(0, len(create_n_var) - 1)
    n_select_var = create_n_var[n_select_var_index]

    if method == 'random':
        remain_index = set()
        while len(remain_index) < n_select_var:
            random_state += 1
            random.seed(random_state)
            index_random = random.randint(0, len(df_data_noisy.iloc[0]) - 1)
            remain_index.add(index_random)

        df_concat = pd.concat([df_data.copy(), df_data_noisy.iloc[:, list(remain_index)]], axis=1)
    elif method == 'Relief':
        df_score=pd.DataFrame()
        for i,e in enumerate(cons.SEEDS):
            data_train_scaled, y_train, categorical_col,numeric_col = preprocess_for_fs(df_data_noisy, labels, e)
            feature_selected=perform_ensemble_fs(data_train_scaled,
                                np.asarray(y_train),
                                'relief',
                                'mean',
                                data_train_scaled.columns,
                                categorical_col,
                                numeric_col,
                                n_boots=5,
                                n_jobs=10)
            column = 'score' + str(i)
            feature_selected.columns = ['names', column]
            try:
                df_score = df_score.merge(feature_selected, on=['names'])
            except:
                df_score = feature_selected
        df_score = sum_score(df_score)
        df_concat = pd.concat([df_data.copy(), df_data_noisy[df_score.names[:int(n_select_var)].values]], axis=1)


    else:
        return ValueError('Method parameter is incorrect, choose between random and Relief')
    return df_concat


def compute_sil_score(dist_matrix, n_clusters, labels, clustering_model):
    if clustering_model.labels_.max() == n_clusters or n_clusters:
        index = np.where(labels != -1)[0]
        positions_to_erase = np.where(labels == -1)[0]
        distances = pd.DataFrame(dist_matrix).drop(positions_to_erase).T
        distances = distances.drop(distances.index[positions_to_erase])
        sil_score = silhouette_score(distances, labels[index], metric='precomputed')
    else:
        sil_score = 0

    return sil_score


def represent_UMAP(dist_matrix, score, baseline_score, labels, save_name):
    if score == baseline_score:
        plot_UMAP(dist_matrix, 20, save_name + '_baseline' + str(np.round(score, 3)), labels,
                  cons.PATH_PROJECT_REPORTS_NOISY_FIGURES)
    else:
        plot_UMAP(dist_matrix, 20, save_name + '_noise_' + str(np.round(score, 3)), labels,
                  cons.PATH_PROJECT_REPORTS_NOISY_FIGURES)


def represent_TSNE(dist_matrix, score, baseline_score, labels, save_name):
    if score == baseline_score:
        plot_TSNE(dist_matrix, save_name + '_baseline_' + str(np.round(score, 3)), labels,
                  cons.PATH_PROJECT_REPORTS_NOISY_FIGURES)
    else:
        plot_TSNE(dist_matrix, save_name + '_noise_' + str(np.round(score, 3)), labels,
                  cons.PATH_PROJECT_REPORTS_NOISY_FIGURES)


def apply_clustering_model(model, dist_matrix):
    v_predicted_clustering = model.fit(dist_matrix.astype('float64')).labels_
    score = silhouette_score(dist_matrix, v_predicted_clustering, metric='precomputed')
    return v_predicted_clustering, score


def compute_clustering(x_data, clustering_method, n_clusters=True, random_state=0, plot= False,baseline_score=-1,
                       save_name=''):
    dist_matrix, data_scaled = preprocess_for_clustering(x_data)
    if clustering_method == 'hdbscan':
        np.random.seed(random_state)
        clustering_model = hdbscan.HDBSCAN(min_samples=int(0.01 * len(x_data)),
                                           min_cluster_size=int(0.05 * len(x_data)), cluster_selection_method='eom',
                                           metric='precomputed')
        v_predicted_clustering = clustering_model.fit(dist_matrix.astype('float64')).labels_
        sil_score = compute_sil_score(dist_matrix, n_clusters, v_predicted_clustering, clustering_model)
        outliers = np.count_nonzero(v_predicted_clustering == -1)
        score = sil_score - outliers / len(v_predicted_clustering)
        detected_clusters = clustering_model.labels_.max() + 1

    elif clustering_method == 'spectral':
        clustering_model = SpectralClustering(n_jobs=-1, n_clusters=n_clusters, eigen_solver='lobpcg',
                                              n_neighbors=int(0.1 * len(x_data)),
                                              affinity='precomputed_nearest_neighbors', random_state=0)
        v_predicted_clustering, score = apply_clustering_model(clustering_model, dist_matrix)
        detected_clusters = n_clusters

    else:
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', affinity='precomputed')
        v_predicted_clustering, score = apply_clustering_model(clustering_model, dist_matrix)
        detected_clusters = n_clusters

    if plot:
        represent_UMAP(dist_matrix, score, baseline_score, v_predicted_clustering, save_name)
        represent_TSNE(dist_matrix, score, baseline_score, v_predicted_clustering, save_name)
    return v_predicted_clustering, score, detected_clusters

def compute_score_with_noise(x_data,labels):
    dist_matrix, data_scaled = preprocess_for_clustering(x_data)
    score = silhouette_score(dist_matrix, labels, metric='precomputed')
    return score

def train_fs_method_by_bootstrap(dict_df, clustering_method, n_clusters, baseline_score, save_name,labels):
    df_concat = dict_df['df_concat']
    logger.info('Clustering with: {} and df-shape: {}'.format(clustering_method, df_concat.shape))
    # v_predicted_clustering, score, detected_cluster = compute_clustering(df_concat.values, clustering_method,
    #                                                                      n_clusters, 0,False,
    #                                                                      baseline_score, save_name)

    score = compute_score_with_noise(df_concat.values, labels)
    dict_fs_result = {
        'id_boot': dict_df['id_boot'],
        'score': score,
        'df_concat': df_concat,
        'df_concat2': dict_df['df_concat2']
    }

    return dict_fs_result

def parse_arguments(parser):
    parser.add_argument('--dataset', default='dermat', type=str)
    parser.add_argument('--categorical_encoding', default='count', type=str)
    parser.add_argument('--clustering_method', default='spectral', type=str)
    parser.add_argument('--n_jobs', default=15, type=int)
    parser.add_argument('--n_boots', default=100, type=int)  # N = 100 candidate augmentations
    parser.add_argument('--n_new_vars', default=3, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--noise_type', default='heterogeneous', type=str)
    return parser.parse_args()


def save_split_results(df_concat, y_label, noise_type, dataset, seed, partition):
    df_concat_with_labels = df_concat.reset_index(drop=True)
    df_concat_with_labels['label'] = list(y_label)
    csv_file_path = str(get_split_dataset_path(dataset, noise_type, seed, partition))
    df_concat_with_labels.to_csv(csv_file_path, index=False)
    logger.info('Saved {} ({} samples, {} features)'.format(csv_file_path, df_concat_with_labels.shape[0],
                                                            df_concat_with_labels.shape[1] - 1))


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser(description='noise generator')
    args = parse_arguments(parser)

    for seed in consts.SEEDS:
        # The augmentation (candidate generation, spectral clustering and Silhouette-based selection) is fitted on
        # the training partition only. The selected configuration (noisy features retained and their per-feature
        # noise parameters, with the same random seed) is then applied unchanged to the test partition.
        x_train, y_train, x_test, y_test = load_split_dataset(args.dataset, 'preprocessed', seed,
                                                              args.categorical_encoding)
        x_test = x_test[x_train.columns]

        df_train_noisy, df_test_noisy = generate_n_noisy_features(x_train.reset_index(drop=True),
                                                                  y_train,
                                                                  n_new_features=args.n_new_vars,
                                                                  clustering_method=args.clustering_method,
                                                                  noise_type=args.noise_type,
                                                                  n_boots=args.n_boots,
                                                                  n_jobs=args.n_jobs,
                                                                  save_name="_".join(
                                                                      [args.dataset, args.noise_type,
                                                                       args.categorical_encoding,
                                                                       'seed', str(seed)]),
                                                                  dataset2=x_test.reset_index(drop=True)
                                                                  )

        save_split_results(df_train_noisy, y_train, args.noise_type, args.dataset, seed, 'train')
        save_split_results(df_test_noisy, y_test, args.noise_type, args.dataset, seed, 'test')

    logger.info('Execution time: {:.1f} s'.format(time.time() - start_time))
