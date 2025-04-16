"""
Author: Yu Chen
Email: yu_chen2000@hust.edu.cn
Website: hustyuchen.github.io
"""


import numpy as np
import random

import pandas as pd


def query_max_uncertainty(y_pred, n_instances=100):
    """
    选择模型最不确定的样本索引，即概率最大项最小的那n_instances个

    Args:
        y_pred: 模型预测的概率值.
        n_instances: 抽取的样本数.

    Returns:
        index：在y_pred中的索引
    """
    y_pred_max = np.max(y_pred, axis=-1)
    return np.argsort(y_pred_max)[:n_instances]



def query_margin_prob(y_pred, n_instances=100):
    """
    选择模型边际概率最小的那n_instances个

    Args:
        y_pred: 模型预测的概率值.
        n_instances: 抽取的样本数.

    Returns:
        index：在y_pred中的索引`
    """
    y_pred_sort = np.sort(y_pred, axis=1)
    y_pred_margin = y_pred_sort[:, 2]-y_pred_sort[:,1]
    return np.argsort(y_pred_margin)[:n_instances]


def query_max_entropy(y_pred, n_instances=100):
    """
    选择模型预测熵最大的那n_instances个

    Args:
        y_pred: 模型预测的概率值.
        n_instances: 抽取的样本数.

    Returns:
        index：在y_pred中的索引
    """
    # just normal (0, 1)
    y_pred = 1/(1 + np.e**(-y_pred))
    y_pred_log = np.log(y_pred)
    entropy = np.sum(-y_pred * y_pred_log, axis=1)

    return np.argsort(entropy)[-n_instances:]




def query_margin_kmeans(y_pred, feature, n_instances=100):
    """
    考虑多样性采样，聚类后从每个簇中选择informative_score最高的样本

    Args:
        y_pred: 模型预测的概率值.
        feature：模型倒数第二层输出值或自编码器提取的特征
        n_instances: 抽取的样本数.

    Returns:
        index：在y_pred中的索引
    """

    y_pred = 1/(1 + np.e**(-y_pred))
    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
    y_pred_log = np.log(y_pred)
    entropy = np.sum(-y_pred * y_pred_log, axis=1)
    Cluster = KMeans(n_clusters=n_instances).fit(feature, sample_weight=entropy-min(entropy)+0.01)

    '''
    from sklearn.decomposition import PCA, KernelPCA
    #pca = PCA(n_components=10, whiten=True).fit(y_pred_second)
    #kpca = KernelPCA(n_components=20, kernel='rbf').fit(y_pred_second)
    #reduced_y_pred_second = kpca.transform(y_pred_second)
    y_pred_log = np.log(y_pred)
    entropy = np.sum(-y_pred * y_pred_log, axis=1)
    np.savetxt('y_pred_margin.npy', y_pred_margin)
    Cluster = KMeans(n_clusters=n_instances).fit(y_pred_second, sample_weight=1-y_pred_margin)
    labels = Cluster.labels_
    '''

    # 分别找到各个簇的样本索引
    #cluster_category = [np.where(labels == i) for i in range(n_instances)]
    # 从每个簇中选择margin最小的样本
    #index = [cluster_category[i][0][np.argmin(y_pred_margin[cluster_category[i]])] for i in range(n_instances)]
    distance = Cluster.transform(feature)
    selected = np.argmin(distance, axis=0)
    return selected


def query_margin_kmeans_pure_diversity(feature, n_instances=100):
    """
    仅考虑样本多样性，根据feature对样本进行聚类，选择聚类中心

    Args:
        feature：模型倒数第二层输出值或自编码器提取的特征
        n_instances: 抽取的样本数.

    Returns:
        index：在y_pred中的索引
    """

    from sklearn.cluster import KMeans
    Cluster = KMeans(n_clusters=n_instances).fit(feature)
    labels = Cluster.labels_
    distance = Cluster.transform(feature)
    selected = np.argmin(distance, axis=0)
    return selected


def query_margin_kmeans_2stages(y_pred, feature, n_instances, beta):
    """
    融合不确定性和多样性的两阶段采样方法，所提方法

    Args:
        y_pred: 模型预测的概率值.
        feature：模型倒数第二层输出值或自编码器提取的特征
        n_instances: 抽取的样本数.
        beta: 预选的样本数量倍数，预选beta*n_instances个样本，再从中选择n_instances个

    Returns:
        index：在y_pred中的索引
    """

    y_pred_sort = np.sort(y_pred, axis=1)
    y_pred_margin = y_pred_sort[:,2] - y_pred_sort[:,1]
    # 边际概率最小的beta*n_instances个样本的索引
    y_pred_margin_bottom_index = np.argsort(y_pred_margin)[:beta*n_instances]

    from sklearn.cluster import KMeans
    Cluster = KMeans(n_clusters=n_instances).fit(feature[y_pred_margin_bottom_index])
    labels = Cluster.labels_
    distance = Cluster.transform(feature[y_pred_margin_bottom_index])
    selected = np.argmin(distance, axis=0)
    return y_pred_margin_bottom_index[selected]

def random_sampleing(feature, n_instances):
    selected_list = [i for i in range(feature.shape[0])]
    random.shuffle(selected_list)
    return selected_list[-n_instances:]


if __name__ == '__main__':
    a = random_sampleing(np.zeros(shape=(100,3)),32)
