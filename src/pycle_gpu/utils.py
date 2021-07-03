"""
Toolbox for miscellaneous methods.
Leon Zheng
"""

import numpy as np
import torch


def compute_assignment(dataset, centroids):
    distances = torch.cdist(dataset, centroids)
    labels = torch.argmin(distances, dim=1)
    return labels


def compute_clusters_from_assignment(dataset, centroids):
    labels = compute_assignment(dataset, centroids)
    clusters = [[] for k in range(len(centroids))]
    for k in range(len(centroids)):
        clusters[k] = dataset[labels == k]
    return clusters, labels


def sum_of_errors(dataset, centroids):
    """Computes the Sum of Errors of some centroids on a dataset, given by
        SE(X,C) = sum_{x_i in X} min_{c_k in C} ||x_i-c_k||_2.

    Arguments:
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - C: (K,d)-numpy array, the K centroids in dimension d

    Returns:
        - SSE: real, the SSE score defined above
    """
    distances = torch.cdist(dataset, centroids)

    return torch.sum(torch.min(distances, dim=1)[0])


def sum_of_squarred_errors(dataset, centroids):
    """Computes the Sum of Errors of some centroids on a dataset, given by
        SE(X,C) = sum_{x_i in X} min_{c_k in C} ||x_i-c_k||_2.

    Arguments:
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - C: (K,d)-numpy array, the K centroids in dimension d

    Returns:
        - SSE: real, the SSE score defined above
    """
    distances = torch.square(torch.cdist(dataset, centroids))
    return torch.sum(torch.min(distances, dim=1)[0])


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def clustering_metrics(features, centroids):
    dic = {}
    # Compute clusters
    clusters, labels = compute_clusters_from_assignment(features, centroids)
    size_clusters = []
    for cluster in clusters:
        size_clusters.append(len(cluster))
    # print(size_clusters)
    size_clusters = np.array(size_clusters)
    dic['size_max'] = float(np.max(size_clusters))
    dic['size_min'] = float(np.min(size_clusters))
    dic['size_mean'] = float(np.mean(size_clusters))

    dic['relative_size_max'] = float(np.max(size_clusters) / len(features))
    dic['relative_size_min'] = float(np.min(size_clusters) / len(features))
    dic['relative_size_mean'] = float(np.mean(size_clusters) / len(features))

    dic['number_near_empty_clusters'] = float(np.sum(size_clusters / len(features) <
                                                     dic['relative_size_mean'] * 0.01))
    dic['number_empty_clusters'] = float(np.sum(size_clusters == 0))

    # dic['entropy_size_clusters'] = float(scipy.stats.entropy(size_clusters / len(features)))
    # dic['entropy_uniform_distribution'] = float(scipy.stats.entropy(dic['relative_size_mean']
    #                                                                 * np.ones(len(features))))
    # dic['relative_entropy_size_clusters'] = dic['entropy_size_clusters'] / dic['entropy_uniform_distribution']

    norms = torch.norm(centroids, dim=1)
    dic['centroids_max_norm'] = float(torch.max(norms).item())
    dic['centroids_mean_norm'] = float(torch.mean(norms).item())
    dic['centroids_min_norm'] = float(torch.min(norms).item())

    # dic['nmi'] = normalized_mutual_info_score(labels, assignment)

    # avg_dist_intra_cluster = []
    # std_dist_intra_cluster = []
    # for i, cluster in enumerate(clusters):
    #     if len(cluster) > 0:
    #         distance_to_centroid = np.linalg.norm(np.array(cluster) - centroids[i], axis=1)
    #         avg_dist_intra_cluster.append(np.mean(distance_to_centroid))
    #         std_dist_intra_cluster.append(np.std(distance_to_centroid))
    # dic['max_intra_cluster_avg_dist'] = float(np.max(avg_dist_intra_cluster))
    # dic['min_intra_cluster_avg_dist'] = float(np.min(avg_dist_intra_cluster))
    # dic['mean_intra_cluster_avg_dist'] = float(np.min(avg_dist_intra_cluster))

    # dic['mean_intra_cluster_std_dist'] = float(np.mean(std_dist_intra_cluster))

    return dic


