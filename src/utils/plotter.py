import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from tensorflow.python.ops.gen_experimental_dataset_ops import list_dataset

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from pathlib import Path
import os
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter
from sklearn.manifold import TSNE
import umap
import seaborn
import utils.consts as consts
from aeon.visualisation import plot_critical_difference

def plot_distributions(df: pd.DataFrame,name:str,color='red'):
    for e in df.columns:
        # Plot distribution using Seaborn
        print(e, np.mean(df[e]), df[e].std(), min(df[e]), max(df[e]), )
        print(df[e].value_counts())
        sns.histplot(df[e], color=color)
        plt.title(e,fontsize=20)
        plt.xlabel('Value')
        plt.ylabel('Distribution')
        plt.savefig(os.path.join(consts.PATH_PROJECT_FIGURES, name + '_' + e + '.png'))
        plt.close()


def plot_barplot_comparison_predictive_performance(list_models, list_scores, flag_save_figure=False):
    cmap = get_cmap('Dark2')
    colors = [cmap(i) for i in np.linspace(0, 1, 7)]

    fig, ax = plt.subplots()
    pct_formatter = FuncFormatter(lambda x, pos: '{:.1f}'.format(x * 100))
    ax.bar(np.arange(8),
           list_scores,
           color=colors,
           tick_label=list_models)
    ax.set_ylim(0.65, 0.80)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Accuracy on the test subset (%)', fontsize=14)
    ax.yaxis.set_major_formatter(pct_formatter)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.subplots_adjust(bottom=0.15)

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'predictive_performance.pdf'.format())))
    else:
        plt.show()
    plt.close()


def plot_heatmap(data,save_path,name):

    fig=seaborn.heatmap(data,xticklabels=False,yticklabels=False)
    plt.savefig(str(os.path.join(save_path, name+'_heatmap.png')), bbox_inches='tight')
    plt.close()


def plot_UMAP(x, nn, name, label, save_path=None):
    colors = ListedColormap(['green', 'blue', 'red', 'yellow', 'purple'])
    clusterable_embedding = umap.UMAP(n_neighbors=nn, n_jobs=3, n_components=2, min_dist=0.25, random_state=0,
                                      init='pca', metric='precomputed').fit_transform(x)
    scatter = plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
                          c=label, s=20, cmap=colors)

    plt.xticks(fontname='serif', fontsize=16)
    plt.yticks(fontname='serif', fontsize=16)
    plt.xlabel('UMAP_1', fontname='serif', fontsize=17)
    plt.ylabel('UMAP_2', fontname='serif', fontsize=17)
    labels = []
    for e in range(len(set(label))):
        labels.append('Cluster ' + str(e))
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.savefig(str(os.path.join(save_path, 'UMAP_' + str(name) + '.png')),
                bbox_inches='tight')
    plt.close()


def plot_TSNE(x, name, label, save_path=None):
    colors = ListedColormap(['green', 'blue', 'red', 'yellow', 'purple'])
    clusterable_embedding = TSNE(n_components=2, init='random', verbose=5, perplexity=100, n_iter=600,
                                 random_state=0, metric='precomputed').fit_transform(x)
    scatter = plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
                          c=label, s=20, cmap=colors)

    plt.xticks(fontname='serif', fontsize=16)
    plt.yticks(fontname='serif', fontsize=16)
    plt.xlabel('TSNE_1', fontname='serif', fontsize=17)
    plt.ylabel('TSNE_2', fontname='serif', fontsize=17)
    labels = []

    for e in range(len(set(label))):
        labels.append('Cluster ' + str(e))
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.savefig(str(os.path.join(save_path, 'TSNE_' + str(name) + '.png')),
                bbox_inches='tight')
    plt.close()


def plot_cvi(cvi, names, name_save, save_path):
    for j in range(len(cvi)):
        figure, axis = plt.subplots(1)
        figure.set_size_inches(3, 5)
        plt.plot(np.arange(2, 10, 1), cvi[j])
        plt.xticks(fontname='serif', fontsize=16)
        plt.yticks(fontname='serif', fontsize=16)
        plt.xlabel('Number of clusters', fontname='serif', fontsize=17)
        plt.ylabel("CVI  " + names[j] + ' score', fontname='serif', fontsize=17)
        plt.savefig(str(os.path.join(save_path, 'CVI_' + names[j] + '_' + name_save + '.png')),
                    bbox_inches='tight')
        plt.close()


def plot_comparison_real_noisy_vars(v_real, v_noisy):
    m_real_noisy = np.hstack((v_real, v_noisy))
    df_real_noisy = pd.DataFrame(m_real_noisy)
    plot_hists_real_noisy_var(df_real_noisy)


def plot_hists_real_noisy_var(df):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 12), sharey=True)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Real vs Noisy", fontsize=14, y=0.95)

    tickers = df.columns.values

    for ticker, ax in zip(tickers, axs.ravel()):
        df_counts = df.loc[:, ticker].value_counts().reset_index()
        sns.barplot(x=ticker, y='count', data=df_counts, ax=ax)
        # ax.bar_label(ax.containers[0])
        ax.set_title(ticker)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.show()


def plot_corr_mixed_matrix(df_corr,
                           csv_filepath=None,
                           figsize=(32, 32),
                           flag_save_figure=False,
                           type_linkage='complete',
                           ):
    df_corr_sorted = perform_cluster_corr(df_corr, inplace=False, type_linkage=type_linkage)

    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=0.9)
    sns.heatmap(df_corr_sorted,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0,
                vmax=1.0,
                annot=False,
                fmt=".2f",
                square=True,
                ax=ax
                )

    plt.tight_layout()

    if flag_save_figure:
        plt.savefig(csv_filepath)
    else:
        plt.show()


def perform_cluster_corr(corr_array, inplace=False, type_linkage='complete'):
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method=type_linkage)
    cluster_distance_threshold = pairwise_distances.max() / 2
    print(cluster_distance_threshold)

    idx_to_cluster_array = sch.fcluster(linkage,
                                        0.9,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]

    return corr_array[idx, :][:, idx]


def compare_distribution_var(df_features, varname_1, varname_2):
    data_var1 = df_features[varname_1]
    data_var2 = df_features[varname_2]

    plt.figure(figsize=(10, 5))
    plt.hist(data_var1, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram {}'.format(varname_1))
    plt.xlabel('values')
    plt.ylabel('frequency')
    plt.show()

    # Crear el gráfico de barras para la Variable Categórica
    # plt.figure(figsize=(8, 5))
    # sns.countplot(data_var2, palette='viridis')
    # plt.title('Gráfico de Barras para Variable Categórica')
    # plt.xlabel('Categoría')
    # plt.ylabel('Frecuencia')
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(data_var2, bins=20, color='blue', alpha=0.7)
    plt.xlabel('values')
    plt.ylabel('frequency')
    plt.show()


def plot_aucroc_with_errorbars(dataset_names, dfs, metric):
    """
    Genera un gráfico de barras de error para AUCROC por modelo y dataset.

    Parámetros:
    - dataset_names: Lista con los nombres de los datasets.
    - dfs: Lista de dos DataFrames [df_aucroc, df_std]
        df_aucroc: valores AUC para cada modelo (columnas) y dataset (filas)
        df_std: valores de desviación estándar correspondientes
    """
    df_aucroc, df_std = dfs
    models = df_aucroc.columns
    x = np.arange(len(dataset_names))
    width = 0.06  # ancho entre puntos por modelo

    plt.figure(figsize=(16, 6))

    markers = ['*', '*', '*', '*','*', '^', '^', '^']
    colors = ['green', 'red', 'blue', 'orange', 'gray', 'deeppink', 'brown', 'black']

    for idx, model in enumerate(models):
        aucs = df_aucroc[model].values
        stds = df_std[model].values
        offset = (idx - len(models)/2) * width + width/2
        plt.errorbar(x + offset, aucs, yerr=stds, fmt=markers[idx],
                     color=colors[idx], capsize=4, label=model)

    plt.xticks(x, dataset_names, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel(metric.upper(), fontsize=25)
    plt.xlabel('Datasets', fontsize=25)
    plt.ylim(0.55, 1.02)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(consts.PATH_PROJECT_FIGURES, f"performance_{metric}.pdf"), bbox_inches="tight")

def plot_results(list_databases, list_models, metric ='auc'):
    df_final_auc=pd.DataFrame()
    df_final_std = pd.DataFrame()
    for model in list_models:
        list_auc=[]
        list_auc_std=[]
        for database in list_databases:
            if model not in ['LM-IGTD-HoNG', 'LM-IGTD-HeNG','LM-IGTD']:
                df =pd.read_csv(os.path.join(consts.PATH_PROJECT_METRICS, 'best_models',f'metrics_classification_{database}.csv'))
                df_filtrado = df[df['model'].str.contains(model.lower(), case=False, na=False) &
                                 df['eval_metric'].str.contains(metric, case=False, na=False)]

                list_auc.append(df_filtrado['mean'].values)
                list_auc_std.append(df_filtrado['std'].values)
            elif model == 'LM-IGTD-HoNG':
                df =pd.read_csv(os.path.join(consts.PATH_PROJECT_METRICS, 'best_models',f'metrics_classification_cnn_homogeneous_{database}.csv'))
                df_filtrado = df[df['eval_metric'].str.contains(metric, case=False, na=False)]
                list_auc.append(df_filtrado['mean'].values)
                list_auc_std.append(df_filtrado['std'].values)
            elif model == 'LM-IGTD-HeNG':
                df = pd.read_csv(os.path.join(consts.PATH_PROJECT_METRICS, 'best_models',
                                              f'metrics_classification_cnn_heterogeneous_{database}.csv'))
                df_filtrado = df[df['eval_metric'].str.contains(metric, case=False, na=False)]
                list_auc.append(df_filtrado['mean'].values)
                list_auc_std.append(df_filtrado['std'].values)
            elif model == 'LM-IGTD':
                df = pd.read_csv(os.path.join(consts.PATH_PROJECT_METRICS, 'best_models',
                                              f'metrics_classification_cnn_preprocessed_{database}.csv'))
                df_filtrado = df[df['eval_metric'].str.contains(metric, case=False, na=False)]
                list_auc.append(df_filtrado['mean'].values)
                list_auc_std.append(df_filtrado['std'].values)
        df_final_auc[model]=list_auc
        df_final_std[model]=list_auc_std

    plot = plot_critical_difference(df_final_auc.values, df_final_auc.columns, test='nemenyi', correction='bonferroni',
                                    alpha=0.1, width=10)
    plot[0].savefig(os.path.join(consts.PATH_PROJECT_FIGURES,f"cd_models_{metric}.pdf"), bbox_inches="tight")

    plot_aucroc_with_errorbars(list_databases, [df_final_auc,df_final_std], metric)

list_databases =  ['saheart', 'ionos', 'german', 'crx',  'diabetes' ]
list_models = ['DT', 'RF', 'SVM', 'LASSO', 'LM-IGTD', 'LM-IGTD-HoNG', 'LM-IGTD-HeNG']

plot_results(list_databases, list_models)
plot_results(list_databases, list_models, 'sensitivity')
plot_results(list_databases, list_models, 'specificity')
plot_results(list_databases, list_models, 'accuracy')