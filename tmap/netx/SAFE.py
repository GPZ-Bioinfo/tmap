import time

import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests

from tmap.tda.utils import verify_metadata, unify_data, parallel_works


def _permutation(data, graph=None, shuffle_by='node'):
    """

    :param data: dynamic shape depending on the by.
    :param graph:
    :param shuffle_by: one of node|sample
    :return: it must be a matrix with node x features shapes.
    :rtype: pd.DataFrame
    """
    if shuffle_by == 'node':
        assert data.shape[0] == len(graph.nodes)
        # permute the node attributes, with the network structure kept
        # inplace change
        p_data = data.apply(lambda col: np.random.permutation(col),
                            axis=0)
        return p_data

    elif shuffle_by == 'sample':
        assert data.shape[0] == graph.rawX.shape[0]
        p_data = data.apply(lambda col: np.random.permutation(col), axis=0)
        p_data = graph.transform_sn(p_data, type='s2n')
        # restrict the speed of shuffle by sample.
        # take double time compared to shuffle_by node.
        return p_data


def convertor(compared_count, n_iter):
    """
    Using 'the number of times' between observed values and shuffled values to calculated SAFE score.
    (Multi-test corrected)
    :param compared_count:
    :param node_data:
    :param n_iter:
    :return:
    """
    min_p_value = 1.0 / (n_iter + 1.0)

    neighborhood_count_df = compared_count

    p_value_df = neighborhood_count_df.div(n_iter)
    p_value_df = p_value_df.where(p_value_df >= min_p_value, min_p_value)

    # todo: allow user to specify a multi-test correction method?
    p_values_fdr_bh = p_value_df.apply(lambda col: multipletests(col, method='fdr_bh')[1], axis=0)
    safe_scores = p_values_fdr_bh.apply(lambda col: np.log10(col) / np.log10(min_p_value), axis=0)

    return safe_scores


def PandC(graph, shuffle_by, _p_data, ori_v, neighborhoods, agg_mode, sub_i, q1, q2,random_seed=None):
    # use independent function to perform permutation.
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed(int(time.time() % 1e6))
    # todo: is it a corrected way?
    for _ in range(sub_i):
        p_data = _permutation(_p_data, graph=graph, shuffle_by=shuffle_by)  # it should provide the raw metadata instead of transformed data.
        p_neighborhood_scores = graph.neighborhood_score(node_data=p_data, neighborhoods=neighborhoods, mode=agg_mode)
        _1 = p_neighborhood_scores.values
        enrich = _1 >= ori_v
        decline = _1 <= ori_v
        q1.append((enrich,decline))
        # adding result
        q2.append(0)
        # adding 0 for counter and stop

def _SAFE(graph, data, n_iter=1000, nr_threshold=0.5, neighborhoods=None, shuffle_by="node", _mode='enrich', agg_mode='sum', num_thread=0, verbose=1,random_seed=None):
    """
    perform SAFE analysis by node permutations

    :param tmap.tda.Graph.Graph graph:
    :param data: dynamic shape depending on the shuffle_obj. Input by ``tmap.netx.SAFE.SAFE_batch``
    :param n_iter: number of permutations
    :param nr_threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles for neighbour.
    :param num_thread: number of threads used to parallel jobs
    :return: return dict with keys of nodes ID, values are normalized and multi-test corrected p values.
    """

    if _mode not in ['enrich', 'decline', 'both']:
        raise SyntaxError('_mode must be one of [enrich , decline]')
    if data.shape[0] == len(graph.sample_names):
        # it means provided metadata is shaped as samples x features, so we need transformed it.
        # be carefull the if/else, do not reverse.
        node_data = graph.transform_sn(data,
                                       type='s2n')
    elif data.shape[0] == graph.nodePos.shape[0]:
        node_data = data.copy()
    else:
        raise Exception("Wrong data passed")
    neighborhoods = graph.get_neighborhoods(nr_threshold=nr_threshold) if neighborhoods is None else neighborhoods
    neighborhood_scores = graph.neighborhood_score(node_data=node_data,
                                                   neighborhoods=neighborhoods,
                                                   mode=agg_mode)
    ori_v = neighborhood_scores.values

    _p_data = node_data.copy() if shuffle_by == 'node' else data.copy()

    neighborhood_enrichments, neighborhood_decline = parallel_works(func=PandC,
                                                                    args=(graph,
                                                                          shuffle_by,
                                                                          _p_data,
                                                                          ori_v,
                                                                          neighborhoods,
                                                                          agg_mode),
                                                                    n_iter=n_iter,
                                                                    num_thread=num_thread,
                                                                    verbose=verbose,
                                                                    random_seed=random_seed)

    neighborhood_enrichments = pd.DataFrame(neighborhood_enrichments,
                                            index=list(graph.nodes),
                                            columns=list(data.columns))
    neighborhood_decline = pd.DataFrame(neighborhood_decline,
                                        index=list(graph.nodes),
                                        columns=list(data.columns))
    safe_scores_enrich = convertor(neighborhood_enrichments,
                                   n_iter=n_iter)
    safe_scores_decline = convertor(neighborhood_decline,
                                    n_iter=n_iter)

    if _mode == 'both':
        return {'enrich': safe_scores_enrich, 'decline': safe_scores_decline}
    elif _mode == 'enrich':
        return safe_scores_enrich
    elif _mode == 'decline':
        return safe_scores_decline


def SAFE_batch(graph, metadata, n_iter=1000, nr_threshold=0.5, neighborhoods=None, shuffle_by="node", _mode='enrich', agg_mode='sum', verbose=1, name=None, num_thread=0, random_seed=None,**kwargs):
    """
    Entry of SAFE analysis
    Map sample meta-data to node associated values (using means),
    and perform SAFE batch analysis for multiple features

    For more information, you should see :doc:`how2work`

    :param tmap.tda.Graph.Graph graph:
    :param np.ndarray/pd.DataFrame metadata:
    :param int n_iter: Permutation times. For some features with skewness values, it should be higher in order to stabilize the resulting SAFE score.
    :param float nr_threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles

    :param neighborhoods:
    :param shuffle_by:
    :param _mode:
    :param agg_mode:
    :param verbose:
    :return: return dict ``{feature: {node_ID:p-values(fdr)} }`` .
    """
    neighborhoods = graph.get_neighborhoods(nr_threshold=nr_threshold) if neighborhoods is None else neighborhoods

    if shuffle_by == 'node':
        meta_data = verify_metadata(graph, metadata, by='node')
    else:
        meta_data = verify_metadata(graph, metadata, by='sample')

    all_safe_scores = _SAFE(graph, meta_data,
                            n_iter=n_iter,
                            nr_threshold=nr_threshold,
                            neighborhoods=neighborhoods,
                            _mode=_mode,
                            agg_mode=agg_mode,
                            shuffle_by=shuffle_by,
                            verbose=verbose,
                            num_thread=num_thread,
                            random_seed=random_seed)

    # record SAFE params
    _params = {'shuffle_by': shuffle_by,
               # '_mode':_mode,
               'agg_mode': agg_mode,
               'nr_threshold': nr_threshold,
               'n_iter': n_iter,
               'name': name}

    if _mode == 'both':
        params = _params.copy()
        params['data'] = all_safe_scores['enrich']
        params['_mode'] = 'enrich'
        graph._add_safe(params)
        params = _params.copy()
        params['data'] = all_safe_scores['decline']
        params['_mode'] = 'decline'
        graph._add_safe(params)
    else:
        params = _params.copy()
        params['data'] = all_safe_scores
        params['_mode'] = _mode
        graph._add_safe(params)
    return all_safe_scores


def get_significant_nodes(graph,
                          safe_scores,
                          nr_threshold=0.5,
                          pvalue=0.05,
                          n_iter=None,
                          SAFE_pvalue=None,
                          r_neighbor=False):
    """
    get significantly enriched/declined nodes (>= threshold)
    Difference between centroides and nodes:
        1. centroides mean the node itself
        2. neighbor_nodes mean the neighbor nodes during SAFE calculation (For advanced usage)
    :param tmap.tda.Graph.Graph graph:
    :param pd.DataFrame safe_scores:
    :param nr_threshold:
    :param pvalue:
    :param n_iter:
    :param SAFE_pvalue:
    :param r_neighbor:
    :return:
    """
    neighborhoods = graph.get_neighborhoods(nr_threshold=nr_threshold)
    safe_scores = unify_data(safe_scores)  # become nodes x features matrix
    if safe_scores.shape[0] != len(graph.nodes):
        safe_scores = safe_scores.T
    assert safe_scores.shape[0] == len(graph.nodes)
    if SAFE_pvalue is None:
        n_iter = graph._SAFE[-1]['n_iter'] if n_iter is None else n_iter  # get last score n_iter
        min_p_value = 1.0 / (n_iter + 1.0)
        SAFE_pvalue = np.log10(pvalue) / np.log10(min_p_value)

    sc_dict = safe_scores.to_dict(orient='dict')

    significant_centroids = {f: [n for n in n2v
                                 if n2v[n] >= SAFE_pvalue] for f,
                                                               n2v in sc_dict.items()}

    if r_neighbor:
        significant_neighbor_nodes = {f: list(set([n for n in nodes
                                                   for n in neighborhoods[n]]))
                                      for f, nodes in significant_centroids.items()}
        return significant_centroids, significant_neighbor_nodes
    else:
        return significant_centroids


def get_SAFE_summary(graph, metadata, safe_scores, n_iter=None, p_value=0.01, nr_threshold=0.5, _output_details=False):
    """
    summary the SAFE scores for feature enrichment results
    Noticeably, we will use the index of data which provided to mapper. So if the index of metadata you passed is different with its, it may raise error.
    :param tmap.tda.Graph.Graph graph:
    :param metadata: [n_samples, n_features]
    :param pd.DataFrame safe_scores: node x features matrix
    :param n_iter:
    :param p_value:
    :param nr_threshold:
    :param _output_details:
    :return:
    """
    # todo: refactor into a SAFE summary class?
    if safe_scores.shape[0] != metadata.shape[1]:
        safe_scores = safe_scores.T
    assert safe_scores.shape[0] == metadata.shape[1]
    # safe_score has been transpose into row is features
    if set(metadata.index) != set(graph.rawX.index):
        print("WARNING!!!! The index of metadata and the index of data which provided to mapper are different."
              "It may raise Errors.")

    # make safe_scores become a matrix with shape like (feature,nodes)

    feature_names = safe_scores.index

    safe_total_score = safe_scores.sum(1)
    safe_significant_centroids, safe_significant_neighbor_nodes = get_significant_nodes(graph,
                                                                                        safe_scores=safe_scores,
                                                                                        pvalue=p_value,
                                                                                        nr_threshold=nr_threshold,
                                                                                        n_iter=n_iter,
                                                                                        r_neighbor=True)

    safe_enriched_nodes_n = {feature: len(node_ids) for feature,
                                                        node_ids in safe_significant_centroids.items()}

    safe_significant_samples = {f: graph.node2sample(nodes) for f,
                                                                nodes in safe_significant_centroids.items()}

    safe_significant_samples_n = {feature: len(sample_names) for feature,
                                                                 sample_names in safe_significant_samples.items()}
    safe_significant_score = {feature: np.sum(safe_scores.loc[feature,
                                                              safe_significant_centroids[feature]])
                              for feature in feature_names}
    if _output_details:
        safe_summary = {'enriched_neighbor_nodes': safe_significant_neighbor_nodes,
                        'enriched_centroids_nodes': safe_significant_centroids,
                        'enriched_score': safe_significant_score, }
        return safe_summary

    # calculate enriched ratios ('enriched abundance' / 'total abundance')
    feature_total = metadata.sum(axis=0)
    remained_features = [feature for feature in feature_names if safe_significant_samples[feature]]

    enriched_abundance_ratio = {feature: np.sum(metadata.loc[safe_significant_samples[feature],
                                                             feature]) / feature_total[feature] if feature in remained_features
    else 0 for feature in feature_names}

    # helper for safe division for integer and divide_by zero
    def _safe_div(x, y):
        if y == 0.0:
            return np.nan
        else:
            return x * 1.0 / y

    enriched_safe_ratio = {feature: _safe_div(safe_significant_score[feature],
                                              safe_total_score[feature])
                           for feature in feature_names}

    safe_summary = pd.DataFrame({'SAFE total score': safe_total_score.to_dict(),
                                 'number of enriched nodes': safe_enriched_nodes_n,
                                 'number of enriched samples': safe_significant_samples_n,
                                 'SAFE enriched score': safe_significant_score,
                                 'enriched abundance ratio': enriched_abundance_ratio,
                                 'enriched SAFE score ratio': enriched_safe_ratio,
                                 })
    safe_summary.index.name = 'name'
    return safe_summary
