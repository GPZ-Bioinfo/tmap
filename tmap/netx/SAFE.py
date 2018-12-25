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
    if not isinstance(all_pairs_dist, dict):
        all_pairs_dist = dict(all_pairs_dist)
    return all_pairs_dist


def construct_node_data(graph, data):
    nodes = graph['nodes']
    if "iloc" in dir(data):
        node_data = {k: data.iloc[v, :].mean() for k, v in nodes.items()}
        node_data = pd.DataFrame.from_dict(node_data, orient='index', columns=data.columns)
    else:
        node_data = {k: np.mean(data[v, :], axis=0) for k, v in nodes.items()}
        node_data = pd.DataFrame.from_dict(node_data, orient='index')
    return node_data


def nodes_neighborhood(all_pairs_dist, graph, threshold=0.5):
    """
    generate neighborhoods from the graph for all nodes
    :param graph:
    :param threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles
    :return: return a dict with keys of nodes, values is a list of tuple (another node id, its sizes).
    """
    # all_pairs_dist = nodes_pairwise_dist(graph)
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
        neighbors.remove(node_id)
        # neighbors = dict([(neighbor_id, node_sizes[neighbor_id]) for neighbor_id in neighbors])
        neighborhoods[node_id] = neighbors
    return neighborhoods


def nodes_neighborhood_score(neighborhoods, node_data, _cal_type="dict"):
    """
    calculate neighborhood scores for each node from node associated data
    :param neighborhoods: result from nodes_neighborhood
    :param node_data: node associated values
    :param _cal_type: hidden parameters. For a big data with too many features(>=100), calculation with pandas will faster than using dict.
    :return: return a dict with keys of center nodes, value is a float
    """

    # weighted neighborhood scores by node size
    if (_cal_type == "auto" and node_data.shape[1] < 100) or _cal_type == "dict":
        neighborhood_scores = {k: np.sum([node_data[n_k] for n_k in neighbors])
                               for k, neighbors in neighborhoods.items()}
    elif (_cal_type == "auto" and node_data.shape[1] >= 100) or _cal_type == "df":
        arr = node_data.values
        neighborhood_scores = {k: arr[neighbors].sum(0)
                               for k, neighbors in neighborhoods.items()}
        neighborhood_scores = pd.DataFrame.from_dict(neighborhood_scores, orient="index", columns=node_data.columns)
        # neighborhood_scores = neighborhood_scores.reindex(node_data.index)
    else:
        if _cal_type not in ["auto", "df", "dict"]:
            exit("_cal_type must be one of auto,df,dict.")
    return neighborhood_scores

def convertor(neighborhood_test,node_data,n_iter,):
    neighborhood_enrichments = pd.DataFrame(neighborhood_test, columns=node_data.columns, index=node_data.index)
    # perform multiple test correction using the 'fdr_bh' method
    min_p_value = 1.0 / (n_iter + 1.0)
    p_value_df = neighborhood_enrichments.div(n_iter)
    p_value_df[p_value_df <= min_p_value] = min_p_value

    # todo: allow user to specify a multi-test correction method?
    p_values_fdr_bh = p_value_df.apply(lambda col: multipletests(col, method='fdr_bh')[1], axis=0)
    safe_scores = p_values_fdr_bh.apply(lambda col: np.log10(col) / np.log10(min_p_value), axis=0)
    safe_scores = safe_scores.to_dict(orient="dict")
    return safe_scores

def SAFE(graph, node_data, n_iter=1000, nr_threshold=0.5, all_dist=None, neighborhoods=None, _cal_type="dict",_mode='enrich',verbose=1):
    """
    perform SAFE analysis by node permutations
    :param graph:
    :param node_data: node associated values (a dictionary)
    :param n_iter: number of permutations
    :param nr_threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles for neighbour.
    :param _cal_type: hidden parameters. For a big data with too many features(>=100), calculation with pandas will faster than using dict.
    :return: return dict with keys of nodes ID, values are normalized and multi-test corrected p values.
    """
    if _mode not in ['enrich','decline','both']:
        print('_mode must be one of [enrich , decline]')
        return
    all_pairs_dist = nodes_pairwise_dist(graph) if all_dist is None else all_dist
    neighborhoods = nodes_neighborhood(all_pairs_dist, graph, threshold=nr_threshold) if neighborhoods is None else neighborhoods

    neighborhood_scores = nodes_neighborhood_score(neighborhoods, node_data=node_data, _cal_type=_cal_type)
    if (_cal_type == "auto" and node_data.shape[1] >= 100) or _cal_type == "df":
        # enrichment (p-value) as a rank in the permutation scores (>=, ordered)
        neighborhood_enrichments = np.zeros(node_data.shape)
        neighborhood_decline = np.zeros(node_data.shape)
        p_data = node_data.copy()
        if verbose == 0:
            iter_obj = range(n_iter)
        else:
            iter_obj = tqdm(range(n_iter))
        for _ in iter_obj:
            # permute the node attributes, with the network structure kept
            # inplace change
            p_data = p_data.apply(lambda col: np.random.permutation(col), axis=0)
            p_neighborhood_scores = nodes_neighborhood_score(neighborhoods, p_data, _cal_type="df")

            neighborhood_enrichments[p_neighborhood_scores >= neighborhood_scores] += 1
            neighborhood_decline[p_neighborhood_scores <= neighborhood_scores] += 1

        safe_scores_enrich = convertor(neighborhood_enrichments,node_data,n_iter)
        safe_scores_decline = convertor(neighborhood_decline, node_data, n_iter)

    elif (_cal_type == "auto" and node_data.shape[1] < 100) or _cal_type == "dict":
        # enrichment (p-value) as a rank in the permutation scores (>=, ordered)
        neighborhood_enrichments = {k: 0 for k in neighborhood_scores.keys()}
        if verbose == 0:
            iter_obj = range(n_iter)
        else:
            iter_obj = tqdm(range(n_iter))
        for _ in iter_obj:
            # permute the node attributes, with the network structure kept
            p_data = dict(zip(node_data.keys(), np.random.permutation(list(node_data.values()))))
            p_neighborhood_scores = nodes_neighborhood_score(neighborhoods, p_data, _cal_type="dict")
            for k in neighborhood_enrichments.keys():
                if p_neighborhood_scores[k] >= neighborhood_scores[k] and _mode == 'enrich':
                    neighborhood_enrichments[k] += 1
                elif p_neighborhood_scores[k] <= neighborhood_scores[k] and _mode == 'decline':
                    neighborhood_enrichments[k] += 1

        # with no multiple test correction
        # min_p_value = 1.0/(n_iter+1.0)
        # # bound the p-value to (0,1]
        # get_p_value = lambda k: max(float(neighborhood_enrichments[k])/n_iter, min_p_value)
        # safe_scores = {k: np.log10(get_p_value(k))/np.log10(min_p_value)
        #                for k in neighborhood_enrichments.keys()}

        # perform multiple test correction using the 'fdr_bh' method
        min_p_value = 1.0 / (n_iter + 1.0)
        nodes_keys = neighborhood_enrichments.keys()
        get_p_value = lambda k: max(float(neighborhood_enrichments[k]) / n_iter, min_p_value)
        p_values = [get_p_value(k) for k in nodes_keys]
        # todo: allow user to specify a multi-test correction method?
        p_values_fdr_bh = multipletests(p_values, method='fdr_bh')[1]
        p_values_fdr_bh = dict(zip(nodes_keys, p_values_fdr_bh))
        safe_scores_enrich = {k: np.log10(p_values_fdr_bh[k]) / np.log10(min_p_value) for k in nodes_keys}
    else:
        if _cal_type not in ["auto", "df", "dict"]:
            print("_cal_type must be one of auto,df,dict.")
            return
    if _mode =='both':
        return safe_scores_enrich,safe_scores_decline
    elif _mode == 'enrich':
        return safe_scores_enrich
    elif _mode == 'decline':
        return safe_scores_decline


def SAFE_batch(graph, meta_data, n_iter=1000, nr_threshold=0.5, shuffle_obj="node", _cal_type="auto",_mode='enrich',verbose=1):
    """
    Map sample meta-data to node associated values (using means),
    and perform SAFE batch analysis for multiple features

    For more information, you should see :doc:`how2work`

    :param dict graph:
    :param np.ndarry/pd.DataFrame meta_data:
    :param int n_iter: Permutation times. For some features with skewness values, it should be higher in order to stabilize the resulting SAFE score.
    :param float nr_threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles
    :return: return dict ``{feature: {node_ID:p-values(fdr)} }`` .
    """
    all_pairs_dist = nodes_pairwise_dist(graph)
    neighborhoods = nodes_neighborhood(all_pairs_dist, graph, threshold=nr_threshold)

    if (_cal_type == "auto" and meta_data.shape[1] >= 100) or _cal_type == "df":
        if meta_data.shape[0] != len(graph["nodes"]):
            node_data = construct_node_data(graph, meta_data)
        else:
            node_data = meta_data
        all_safe_scores = SAFE(graph, node_data, n_iter=n_iter, nr_threshold=nr_threshold, all_dist=all_pairs_dist, neighborhoods=neighborhoods, _cal_type="df",_mode=_mode,verbose=verbose)
    elif (_cal_type == "auto" and meta_data.shape[1] < 100) or _cal_type == "dict":
        nodes = graph['nodes']
        all_safe_scores = {}
        if "columns" not in dir(meta_data):
            meta_data = pd.DataFrame(meta_data)
        if verbose == 0:
            iter_obj = meta_data.columns
        else:
            iter_obj = tqdm(meta_data.columns)
        for feature in iter_obj:
            if meta_data.shape[0] != len(graph["nodes"]):
                node_data = {k: meta_data.iloc[v, meta_data.columns.get_loc(feature)].mean() for k, v in nodes.items()}
            else:
                node_data = meta_data.to_dict("index")
            safe_scores = SAFE(graph, node_data, n_iter=n_iter, nr_threshold=nr_threshold, all_dist=all_pairs_dist, neighborhoods=neighborhoods, _cal_type="dict",_mode=_mode,verbose=verbose)
            all_safe_scores[feature] = safe_scores
    else:
        if _cal_type not in ["auto", "df", "dict"]:
            exit("_cal_type must be one of auto,df,dict.")
    return all_safe_scores


def SAFE_single(graph, sample_data, n_iter=1000, nr_threshold=0.5):
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
    safe_scores = SAFE(graph, node_data, n_iter=n_iter, nr_threshold=nr_threshold, _cal_type="dict")
    return safe_scores


def get_enriched_nodes(safe_scores, threshold, graph, centroids=False):
    """
    get significantly enriched nodes (>= threshold)
    :param safe_scores:
    :param threshold:
    :return:
    """
    all_pairs_dist = nodes_pairwise_dist(graph)
    neighborhoods = nodes_neighborhood(all_pairs_dist, graph, threshold=threshold)
    if 'columns' in dir(safe_scores):
        node_ids = safe_scores.columns
    else:
        safe_scores = pd.DataFrame.from_dict(safe_scores, orient='index')
        node_ids = safe_scores.columns


    enriched_centroides = {feature: list(node_ids[safe_scores.loc[feature, :] >= threshold]) for feature in
                safe_scores.index}
    enriched_nodes = {f:list(set([n for n in nodes for n in neighborhoods[n]])) for f,nodes in enriched_centroides.items()}
    if centroids:
        return enriched_centroides,enriched_nodes
    else:
        return enriched_nodes


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


def get_SAFE_summary(graph, meta_data, safe_scores, n_iter_value, p_value=0.01, _output_details=False):
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

    min_p_value = 1.0 / (n_iter_value + 1.0)
    threshold = np.log10(p_value) / np.log10(min_p_value)
    if isinstance(safe_scores, dict):
        safe_scores = pd.DataFrame.from_dict(safe_scores, orient='index')
    else:
        if safe_scores.index != meta_data.columns:
            safe_scores = safe_scores.T
    feature_names = safe_scores.index

    safe_total_score = safe_scores.sum(1)

    safe_enriched_centroides,safe_enriched_nodes = get_enriched_nodes(safe_scores=safe_scores, threshold=threshold,graph=graph,centroids=True)
    safe_enriched_nodes_n = {feature: len(node_ids) for feature, node_ids in safe_enriched_nodes.items()}
    if meta_data.shape[0] != len(graph["nodes"]):
        # if input meta_data is (nodes,features) shape.
        safe_enriched_samples = get_enriched_samples(enriched_nodes=safe_enriched_nodes, nodes=graph['nodes'])
        safe_enriched_samples_n = {feature: len(sample_ids) for feature, sample_ids in safe_enriched_samples.items()}
    else:
        safe_enriched_samples_n = {feature: "Unknown" for feature, sample_ids in safe_enriched_nodes.items()}

    safe_enriched_score = {feature: np.sum(safe_scores.loc[feature, safe_enriched_centroides[feature]])
                           for feature in feature_names}

    if _output_details:
        safe_summary = {'enriched_nodes': safe_enriched_nodes,
                        'enriched_score': safe_enriched_score, }
        return safe_summary

    # calculate enriched ratios ('enriched abundance' / 'total abundance')
    feature_abundance = meta_data.sum(axis=0)

    if meta_data.shape[0] != len(graph["nodes"]):
        enriched_abundance_ratio = {feature: np.sum(meta_data.iloc[safe_enriched_samples[feature], meta_data.columns.get_loc(feature)]) / feature_abundance[feature]
                                    for feature in feature_names}
    else:
        enriched_abundance_ratio = \
            {feature: np.sum(meta_data.iloc[safe_enriched_nodes[feature], meta_data.columns.get_loc(feature)]) / feature_abundance[feature]
             for feature in feature_names}

    # helper for safe division for integer and divide_by zero
    def _safe_div(x, y):
        if y == 0.0:
            return np.nan
        else:
            return x * 1.0 / y

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
