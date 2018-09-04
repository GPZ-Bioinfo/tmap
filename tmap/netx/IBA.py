from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


def construct_weighted_node_data(graph, data):
    nodes = graph['nodes']
    if "iloc" in dir(data):
        node_data = {k: data.iloc[v, :].mean() for k, v in nodes.items()}
    else:
        node_data = {k: np.mean(data[v, :], axis=0) for k, v in nodes.items()}
    node_data = pd.DataFrame.from_dict(node_data, orient='index')
    return node_data


def single_are(node_data,feature,mask_array,True_vals):
    one_di_data = np.abs(node_data.loc[:, feature].values - node_data.loc[:, feature].values.reshape(-1, 1))
    fpr, tpr, threshold = roc_curve(1 - True_vals, one_di_data[mask_array])
    auc_vals = auc(fpr, tpr, reorder=True)
    return auc_vals,(fpr, tpr, threshold),feature

def batch_area(graph,X,n_threads=5):
    """
    areas_dict,roc_curve_dict = batch_area(graph,X)
    data = []
    for feature in roc_curve_dict.keys():
        xs,ys,texts = roc_curve_dict[feature]
        data.append(go.Scatter(x=xs,y=ys,name=feature,text=texts))
    plotly.offline.plot(data)

    :param graph:
    :param X:
    :return:
    """
    node_data = construct_weighted_node_data(graph, X)
    rng = np.arange(node_data.shape[0])
    mask_array = rng[:, None] < rng
    # upper triangle

    True_vals = graph["adj_matrix"].fillna(0).values[mask_array]
    areas_dict = {}
    roc_curve_dict = {}
    results = []
    if n_threads != 1:
        pool = Pool(processes=n_threads)
        for feature in X.columns:
            result = pool.apply_async(single_are,(node_data,feature,mask_array,True_vals))
            results.append(result)
        for i in tqdm(results):
            i.wait()
        for i in results:
            if i.ready():
                if i.successful():
                    auc_vals, roc_vals, feature = i.get()
                    areas_dict[feature] = auc_vals
                    roc_curve_dict[feature] = roc_vals
    else:
        for feature in tqdm(X.columns):
            auc_vals,roc_vals, feature = single_are(node_data,feature, mask_array, True_vals)
            areas_dict[feature] = auc_vals
            roc_curve_dict[feature] = roc_vals
    return areas_dict,roc_curve_dict

def shuffle_area(graph,X,n_shuffle=1000,feature=None):
    """
    p_counts,results = shuffle_area(graph,X,"Bacteroides")
    :param graph:
    :param X:
    :param feature:
    :param n_shuffle:
    :return:
    """
    node_data = construct_weighted_node_data(graph, X)
    rng = np.arange(node_data.shape[0])
    mask_array = rng[:, None] < rng
    # upper triangle

    True_vals = graph["adj_matrix"].fillna(0).values[mask_array]
    areas_dict = {}
    for feature in tqdm(X.columns):
        auc_vals, roc_vals, feature = single_are(node_data, feature, mask_array, True_vals)
        areas_dict[feature] = auc_vals

    count_big = defaultdict(int)
    results = defaultdict(list)
    for _ in tqdm(range(n_shuffle)):
        tmp = node_data.T
        tmp = tmp.apply(lambda x: np.random.permutation(x), axis=1, result_type="broadcast")
        if feature:
            iter_feas = feature
        else:
            iter_feas = list(node_data.columns)
        for feature in iter_feas:
            auc_vals, roc_vals, feature = single_are(tmp.T, feature, mask_array, True_vals)
            if auc_vals >= areas_dict[feature]:
                count_big[feature] += 1
                #results[feature].append()
            results[feature].append(auc_vals)
    return count_big,results
