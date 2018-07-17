import networkx as nx
import pandas as pd
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm import tqdm


def nodes_pairwise_dist(graph):
    """
    get all pairwise node distance, including self-distance of 0
    :param graph:
    :return: return a nested dictionary (dict of dict) of pairwise distance
    """
    G = nx.Graph()
    G.add_nodes_from(graph['nodes'].keys())
    G.add_edges_from(graph['edges'])
    all_pairs_dist = nx.all_pairs_shortest_path_length(G)
    return all_pairs_dist


def nodes_neighborhood(graph, threshold=0.5):
    """
    generate neighborhoods from the graph for all nodes
    :param graph:
    :param threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles
    :return: return a dict with keys of nodes, values is a list of tuple (another node id, its sizes).
    """
    all_pairs_dist = nodes_pairwise_dist(graph)
    node_sizes = dict(zip(graph['node_keys'], graph['node_sizes']))

    # generate all pairwise shortest path length (duplicated!!! but is OK for percentile statistics)
    all_length = [_ for it in all_pairs_dist.values() for _ in it.values()]
    # remove self-distance (that is 0)
    all_length = [_ for _ in all_length if _ > 0]
    length_threshold = np.percentile(all_length, threshold)
    # print('Maximum path length threshold is set to be %s' % (length_threshold,))

    neighborhoods = {}
    for node_id in graph['nodes']:
        pairs = all_pairs_dist[node_id]
        # node neighborhood include also the center node
        # neighbors = [n for n, l in pairs.items() if (l <= length_threshold) and (n != node_id)]
        neighbors = [n for n, l in pairs.items() if l <= length_threshold]
        neighbors = [(neighbor_id, node_sizes[neighbor_id]) for neighbor_id in neighbors]
        neighborhoods[node_id] = neighbors
    return neighborhoods


def nodes_neighborhood_score(neighborhoods, node_data):
    """
    calculate neighborhood scores for each node from node associated data
    :param neighborhoods: result from nodes_neighborhood
    :param node_data: node associated values
    :return: return a dict with keys of center nodes, value is a float
    """
    # weighted neighborhood scores by node size
    neighborhood_scores = {k: np.sum([node_data[neighbor[0]]*neighbor[1] for neighbor in neighbors])
                           for k, neighbors in neighborhoods.items()}
    return neighborhood_scores


def SAFE(graph, node_data, n_iter=1000, threshold=0.5):
    """
    perform SAFE analysis by node permutations
    :param graph:
    :param node_data: node associated values (a dictionary)
    :param n_iter: number of permutations
    :param threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles
    :return: return dict with keys of center nodes, values are normalized and multi-test corrected p values.
    """

    neighborhoods = nodes_neighborhood(graph, threshold=threshold)
    neighborhood_scores = nodes_neighborhood_score(neighborhoods, node_data=node_data)

    # enrichment (p-value) as a rank in the permutation scores (>=, ordered)
    neighborhood_enrichments = {k: 0 for k in neighborhood_scores.keys()}
    for _ in range(n_iter):
        # permute the node attributes, with the network structure kept
        p_data = dict(zip(node_data.keys(), np.random.permutation(list(node_data.values()))))
        p_neighborhood_scores = nodes_neighborhood_score(neighborhoods, p_data)
        for k in neighborhood_enrichments.keys():
            if p_neighborhood_scores[k] >= neighborhood_scores[k]:
                neighborhood_enrichments[k] += 1

    # with no multiple test correction
    # min_p_value = 1.0/(n_iter+1.0)
    # # bound the p-value to (0,1]
    # get_p_value = lambda k: max(float(neighborhood_enrichments[k])/n_iter, min_p_value)
    # safe_scores = {k: np.log10(get_p_value(k))/np.log10(min_p_value)
    #                for k in neighborhood_enrichments.keys()}

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


def SAFE_batch(graph, meta_data, n_iter=1000, threshold=0.5):
    """
    map sample meta-data to node associated values (using means),
    and perform SAFE batch analysis for multiple features
    :param graph:
    :param meta_data:
    :param n_iter:
    :param threshold:
    :return:
    """
    nodes = graph['nodes']
    all_safe_scores = {}
    for feature in tqdm(meta_data.columns):
        node_data = {k: meta_data.iloc[v, meta_data.columns.get_loc(feature)].mean() for k, v in nodes.items()}
        safe_scores = SAFE(graph, node_data, n_iter=n_iter, threshold=threshold)
        all_safe_scores[feature] = safe_scores
    return all_safe_scores


def SAFE_single(graph, sample_data, n_iter=1000, threshold=0.5):
    """
    map sample meta-data to node associated values (using means),
    and perform SAFE analysis for a single feature
    :param graph:
    :param sample_data:
    :param n_iter:
    :param threshold:
    :return:
    """
    nodes = graph['nodes']
    node_data = {k: np.mean([sample_data[idx] for idx in v]) for k, v in nodes.items()}
    safe_scores = SAFE(graph, node_data, n_iter=n_iter, threshold=threshold)
    return safe_scores


def get_enriched_nodes(safe_scores, threshold):
    """
    get significantly enriched nodes (>= threshold)
    :param safe_scores:
    :param threshold:
    :return:
    """
    node_ids = safe_scores.columns
    return {feature: list(node_ids[safe_scores.loc[feature,:] >= threshold]) for feature in safe_scores.index}


def get_enriched_samples(enriched_nodes, nodes):
    """
    get significantly enriched samples (samples in enriched nodes)
    there are overlapped samples between nodes, and should be deduplicated
    :param enriched_nodes:
    :param nodes:
    :return:
    """
    return {feature: list(set([sample_id for node_id in node_ids
                               for sample_id in nodes[node_id]]))
            for feature, node_ids in enriched_nodes.items()}


def get_SAFE_summary(graph, meta_data, safe_scores, n_iter_value, p_value=0.01,_output_details=False):
    """
    summary the SAFE scores for feature enrichment results
    :param graph:
    :param meta_data: [n_samples, n_features]
    :param safe_scores: a feature dictionary of node scores
    :param n_iter_value:
    :param p_value:
    :return:
    """
    # todo: refactor into a SAFE summary class?

    min_p_value = 1.0 / (n_iter_value+1.0)
    threshold = np.log10(p_value) / np.log10(min_p_value)
    safe_scores = pd.DataFrame.from_dict(safe_scores, orient='index')
    feature_names = safe_scores.index

    safe_total_score = safe_scores.apply(lambda x: np.sum(x), axis=1)

    safe_enriched_nodes = get_enriched_nodes(safe_scores=safe_scores, threshold=threshold)
    safe_enriched_nodes_n = {feature: len(node_ids) for feature, node_ids in safe_enriched_nodes.items()}
    safe_enriched_samples = get_enriched_samples(enriched_nodes=safe_enriched_nodes, nodes=graph['nodes'])
    safe_enriched_samples_n = {feature: len(sample_ids) for feature, sample_ids in safe_enriched_samples.items()}
    safe_enriched_score = {feature: np.sum(safe_scores.loc[feature, safe_enriched_nodes[feature]])
                           for feature in feature_names}

    if _output_details:
        safe_summary = {'enriched_nodes':safe_enriched_nodes,
                        'enriched_score':safe_enriched_score,
                        }
        return safe_summary

    # calculate enriched ratios ('enriched abundance' / 'total abundance')
    feature_abundance = meta_data.sum(axis=0)
    enriched_abundance_ratio = \
        {feature: np.sum(meta_data.iloc[safe_enriched_samples[feature], meta_data.columns.get_loc(feature)])/feature_abundance[feature]
         for feature in feature_names}

    # helper for safe division for integer and divide_by zero
    def _safe_div(x, y):
        if y == 0.0:
            return np.nan
        else:
            return x*1.0/y

    enriched_safe_ratio = {feature: _safe_div(safe_enriched_score[feature], safe_total_score[feature])
                           for feature in feature_names}

    safe_summary = pd.DataFrame({'SAFE total score': safe_total_score.to_dict(),
                                 'number of enriched nodes': safe_enriched_nodes_n,
                                 'number of enriched samples': safe_enriched_samples_n,
                                 'SAFE enriched score': safe_enriched_score,
                                 'enriched abundance ratio': enriched_abundance_ratio,
                                 'enriched SAFE score ratio': enriched_safe_ratio,
                                 })
    safe_summary.index.name = 'name'
    return safe_summary

