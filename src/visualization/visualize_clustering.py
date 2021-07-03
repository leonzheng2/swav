import pandas
import argparse
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path

from src.pycle_gpu.utils import compute_clusters_from_assignment


def get_parser():
    parser = argparse.ArgumentParser(description='Read a results dataframe, and visualize the clustering')
    parser.add_argument("dataframe_path")
    parser.add_argument("figures_dir")
    parser.add_argument("--filter_path")
    return parser

def filter_centroids(pd, filter_dict):
    assert isinstance(pd, pandas.DataFrame)
    assert isinstance(filter_dict, dict)
    for attribute, values_list in filter_dict.items():
        pd = pd[pd[attribute] in values_list]
    return pd

def pca_projection(features, nb_dimension=2):
    # TODO do PCA procjection
    pass


def main(args):
    dataframe = pandas.read_csv(args.dataframe_path, index_col='id')
    # TODO complete the filtering

    for exp_id, row in dataframe.iterrows():
        print(f"Reading {exp_id}...")
        features_path = row['processed_path']
        centroids_path = row['output_path']
        nb_clusters = row['nb_clusters']
        ratio = row['ratio']
        freq_matrix = row['frequency_matrix']
        algo = row['algo']

        # Load numpy arrays
        features = np.load(features_path)
        centroids = np.load(centroids_path)

        # Random projection
        proj_dim = np.random.choice(features.shape[1], 2, replace=False)
        dim_1 = proj_dim[0]
        dim_2 = proj_dim[1]

        # Save directory
        features_id = Path(features_path).stem
        save_dir = Path(args.figures_dir) / features_id / f"nb_clusters-{nb_clusters}" / f"freq_mat-{freq_matrix}" / f"ratio-{ratio}" / f"algo-{algo}"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Plot
        plot_cluster_assignments_and_histogram(exp_id, features, centroids, save_dir / f"{exp_id}.png", dim_1, dim_2)


def plot_cluster_assignments_and_histogram(title, features, centroids, save_path, dim_1, dim_2):
    clusters, _ = compute_clusters_from_assignment(features, centroids)
    features_np = features.cpu().numpy()
    centroids_np = centroids.cpu().numpy()
    # for cluster in clusters:
    #     print(len(cluster))

    fig, axs = plt.subplots(1, 2, figsize=(10 * 2, 10))
    fig.suptitle(title)
    axs[0].hist2d(features_np[:, dim_1], features_np[:, dim_2], bins=100)
    axs[0].scatter(centroids_np[:, dim_1], centroids_np[:, dim_2], c='red', alpha=1)
    axs[0].set_xlabel(dim_1)
    axs[0].set_ylabel(dim_2)
    for cluster in clusters:
        # print(cluster)
        cluster_np = cluster.cpu().numpy()
        axs[1].scatter(cluster_np[:, dim_1], cluster_np[:, dim_2], s=1, alpha=0.15)
    axs[1].scatter(centroids_np[:, dim_1], centroids_np[:, dim_2], c='red', alpha=1)
    axs[1].set_xlabel(dim_1)
    axs[1].set_ylabel(dim_2)

    plt.savefig(save_path)


def plot_cluster_histogram(title, features, centroids, save_path, dim_1, dim_2):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.hist2d(features[:, dim_1], features[:, dim_2], bins=100)
    print(centroids[:, dim_1], centroids[:, dim_2])
    plt.scatter(centroids[:, dim_1], centroids[:, dim_2], c='red', alpha=1)
    plt.xlabel(dim_1)
    plt.ylabel(dim_2)
    plt.savefig(save_path)


def plot_cluster_scatter(title, features, centroids, save_path, dim_1=0, dim_2=1):
    """
    Plot clustering figure. Put each data point as a small dot in the dimension given by dim_1 and dim_2.
    Put also centroids.
    :param title: string
    :param features: numpy array
    :param centroids: numpy array
    :param dim_1: int
    :param dim_2: int
    :param save_path: path
    :return:
    """
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.scatter(features[:, dim_1], features[:, dim_2], s=1, alpha=0.15)
    plt.scatter(centroids[:, dim_1], centroids[:, dim_2])
    # plt.legend(["Data", "Centroids"])
    plt.savefig(save_path)

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
