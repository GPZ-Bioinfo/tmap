from .SAFE import construct_weighted_node_data
import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm import  tqdm

import warnings;warnings.filterwarnings('ignore')

def connectivity(graph,node_data):
    n_nodes = len(graph["nodes"])

    adj_matrix = graph["adj_matrix"]
    square_sum = np.sum(np.square(node_data))

    connectivity_feas = {}
    for fea in node_data.columns:
        multiple_matrix = np.multiply(node_data.loc[:, fea].values, node_data.loc[:, fea].values.reshape(-1, 1))
        raw_connectivity = np.sum(multiple_matrix[adj_matrix.isna().values])/square_sum[fea]
        connectivity_feas[fea] = raw_connectivity * (n_nodes/(n_nodes-1))
    return connectivity_feas
def SAFE2(graph, data, n_iter=1000):
    """
    perform SAFE analysis by node permutations
    :param graph:
    :param node_data: node associated values (a dictionary)
    :param n_iter: number of permutations
    :param threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles
    :return: return dict with keys of nodes ID, values are normalized and multi-test corrected p values.
    """
    node_data = construct_weighted_node_data(graph,data)
    neighborhood_scores = connectivity(graph, node_data)

    # enrichment (p-value) as a rank in the permutation scores (>=, ordered)
    neighborhood_enrichments = {k: 0 for k in neighborhood_scores.keys()}
    for _ in tqdm(range(n_iter),total=n_iter):
        # permute the node attributes, with the network structure kept
        p_data = node_data.apply(lambda col:np.random.permutation(col),axis=0)
        p_neighborhood_scores = connectivity(graph, p_data)
        for k in neighborhood_enrichments.keys():
            if p_neighborhood_scores[k] >= neighborhood_scores[k]:
                neighborhood_enrichments[k] += 1

    # perform multiple test correction using the 'fdr_bh' method
    min_p_value = 1.0/(n_iter+1.0)
    nodes_keys = neighborhood_enrichments.keys()
    get_p_value = lambda k: max(float(neighborhood_enrichments[k])/n_iter, min_p_value)
    p_values = [get_p_value(k) for k in nodes_keys]
    # todo: allow user to specify a multi-test correction method?
    p_values_fdr_bh = multipletests(p_values, method='fdr_bh')[1]
    p_values_fdr_bh = dict(zip(nodes_keys, p_values_fdr_bh))
    safe_scores = {k: np.log10(p_values_fdr_bh[k])/np.log10(min_p_value) for k in nodes_keys}

    return safe_scores

