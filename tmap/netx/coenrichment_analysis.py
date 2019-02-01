import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as scs
from tqdm import tqdm

from tmap.netx.SAFE import get_enriched_nodes


def get_component_nodes(graph, enriched_nodes):
    """
    Given a list of enriched nodes which comes from ``get_enriched_nodes``. Normally it the enriched_centroid instead of the others nodes around them.
    :param list enriched_nodes: list of nodes ID which is enriched with given feature and threshold.
    :param graph:
    :return: A nested list which contain multiple list of nodes which is not connected to each other.
    """
    sub_nodes = list(enriched_nodes)
    sub_edges = [edge for edge in graph['edges'] if edge[0] in sub_nodes and edge[1] in sub_nodes]

    G = nx.Graph()
    G.add_nodes_from(sub_nodes)
    G.add_edges_from(sub_edges)

    comp_nodes = [nodes for nodes in nx.algorithms.components.connected_components(G)]
    return comp_nodes


def is_enriched(s1, s2, s3, s4):
    """
    Accessory function for further determine whether it is a co-enrichment after fisher-exact test.
    """
    total_1 = len(s1) + len(s2)
    total_2 = len(s3) + len(s4)
    if total_1 == 0 or total_2 == 0:
        return False
    if len(s1) / total_1 > len(s3) / total_2:
        return True
    else:
        return False


def coenrichment_for_nodes(graph, nodes, fea, enriched_centroid, SAFE_pvalue=None, safe_scores=None, _filter=True, mode='both'):
    """
    Coenrichment main function
    With given feature and its enriched nodes, we could construct a contingency table when we comparing to other feature and its enriched nodes.
    For statistical test association between different features, fisher-exact test was applied to each constructed contingency table. Fisher-exact test only consider the association between two classifications instead of the ratio. For coenrichment, we also implement a accessory function called ``is_enriched`` to further judge that if the ratio of enrichment is bigger than the non-enrichement.

    Besides the global co-enrichment, several local enrichment were observed. With ``networkx`` algorithm for finding component, we could extract each local enrichment nodes from global enrichment.

    Because of the complex combination between comparison of local enrichment, two different contingency table was constructed.

    The contingency table which compare different enriched nodes among components and enriched or non-enriched nodes of other features shown below.

    ======================== ================================= =================================
    fea                      fea this comp enriched nodes      fea other comp enriched nodes
    ======================== ================================= =================================
    o_f_enriched_nodes       s1                                s2
    o_f_non-enriched_nodes   s3                                s4
    ======================== ================================= =================================

    The other contingency table which compare enriched nodes within specific component or non-enriched nodes and enriched or non-enriched nodes of other features shown below.

    ======================== ================================= =================================
    fea                      fea this comp enriched nodes      fea non-enriched nodes
    ======================== ================================= =================================
    o_f_enriched_nodes       s1                                s2
    o_f_non-enriched_nodes   s3                                s4
    ======================== ================================= =================================

    For convenient calculation, three different mode [both|global|local] could be choose.

    Output objects contain two different kinds of dictionary. One dict is recording the correlation information between different features, the other is recording the raw contingency table info between each comparsion which is a list of nodes representing s1,s2,s3,s4.

    If mode equals to 'both', it will output global correlative dict, local correlative, metainfo.
    If mode equals to 'global', it will output local correlative, metainfo.
    If mode equals to 'local', it will output global correlative dict, metainfo.

    For global correlative, keys are compared features and values are tuples (oddsratio, pvalue) from fisher-exact test.
    For local correlative, keys are tuples of (index of components, size of components, features) and values are as same as global correlative.
    The remaining metainfo is a dictionary shared same key to global/local correlative but contains contingency table info.

    :param graph: tmap constructed graph
    :param list nodes: a list of nodes you want to process from specific feature
    :param str fea: feature name which doesn't need to exist at enriched_centroid
    :param dict enriched_centroid: enriched_centroide output from ``get_enriched_nodes``
    :param float threshold: None or a threshold for SAFE score. If it is a None, it will not filter the nodes.
    :param dict safe_scores: A dictionary which store SAFE_scores for filter the nodes. If you want to filter the nodes, it must simultaneously give safe_scores and threshold.
    :param str mode: [both|global|local]
    :return:
    """

    if mode not in ['both', 'global', 'local']:
        print("Wrong syntax input, mode should be one of [both|global|local]")
        return

    total_nodes = set(list(graph["nodes"].keys()))
    comp_nodes = get_component_nodes(graph, nodes)
    metainfo = {}
    global_correlative_feas = {}
    sub_correlative_feas = {}
    if SAFE_pvalue is not None and safe_scores is not None:
        fea_enriched_nodes = set([_ for _ in nodes if safe_scores.get(_, 0) >= SAFE_pvalue])
    else:
        fea_enriched_nodes = set(nodes[::])
    fea_nonenriched_nodes = total_nodes.difference(fea_enriched_nodes)
    metainfo[fea] = fea_enriched_nodes, comp_nodes
    for o_f in enriched_centroid.keys():
        if o_f != fea:
            o_f_enriched_nodes = set(enriched_centroid[o_f])
            o_f_nonenriched_nodes = total_nodes.difference(o_f_enriched_nodes)
            if mode == 'both' or 'global':
                # contingency table
                #                           fea enriched nodes, fea non-enriched_nodes
                # o_f_enriched_nodes           s1                        s2
                # o_f_non-enriched_nodes       s3                        s4
                s1 = o_f_enriched_nodes.intersection(fea_enriched_nodes)
                s2 = o_f_enriched_nodes.intersection(fea_nonenriched_nodes)
                s3 = o_f_nonenriched_nodes.intersection(fea_enriched_nodes)
                s4 = o_f_nonenriched_nodes.intersection(fea_nonenriched_nodes)
                oddsratio, pvalue = scs.fisher_exact([[len(s1), len(s2)],
                                                      [len(s3), len(s4)]])
                if _filter:
                    if pvalue <= 0.05 and is_enriched(s1, s2, s3, s4):
                        global_correlative_feas[o_f] = (oddsratio, pvalue)
                        metainfo[o_f] = (s1, s2, s3, s4)
                else:
                    global_correlative_feas[o_f] = (oddsratio, pvalue)
                    metainfo[o_f] = (s1, s2, s3, s4)

            if mode == 'both' or 'local':
                for idx, nodes in enumerate(comp_nodes):
                    # contingency table
                    #                           fea this comp enriched nodes, fea other comp enriched nodes
                    # o_f_enriched_nodes           s1                          s2
                    # o_f_non-enriched_nodes       s3                          s4
                    nodes = set(nodes)
                    _s1 = o_f_enriched_nodes.intersection(nodes)
                    _s2 = o_f_enriched_nodes.intersection(fea_enriched_nodes.difference(nodes))
                    _s3 = o_f_nonenriched_nodes.intersection(nodes)
                    _s4 = o_f_nonenriched_nodes.intersection(fea_enriched_nodes.difference(nodes))
                    oddsratio, pvalue1 = scs.fisher_exact([[len(_s1), len(_s2)],
                                                           [len(_s3), len(_s4)]])
                    # contingency table
                    #                           fea this comp enriched nodes, fea non-enriched nodes
                    # o_f_enriched_nodes           s1                          s2
                    # o_f_non-enriched_nodes       s3                          s4
                    s1 = o_f_enriched_nodes.intersection(nodes)
                    s2 = o_f_enriched_nodes.intersection(fea_nonenriched_nodes)
                    s3 = o_f_nonenriched_nodes.intersection(nodes)
                    s4 = o_f_nonenriched_nodes.intersection(fea_nonenriched_nodes)
                    oddsratio, pvalue2 = scs.fisher_exact([[len(s1), len(s2)],
                                                           [len(s3), len(s4)]])
                    if _filter:
                        if pvalue1 <= 0.05 and pvalue2 <= 0.05 and is_enriched(s1, s2, s3, s4) and is_enriched(_s1, _s2, _s3, _s4):
                            sub_correlative_feas[(idx, len(nodes), o_f)] = (pvalue1, pvalue2)
                            metainfo[(idx, len(nodes), o_f)] = (_s1, _s2, s2)
                    else:
                        sub_correlative_feas[(idx, len(nodes), o_f)] = (pvalue1, pvalue2)
                        metainfo[(idx, len(nodes), o_f)] = (_s1, _s2, s2)

    if mode == 'both':
        return global_correlative_feas, sub_correlative_feas, metainfo
    elif mode == 'global':
        return global_correlative_feas, metainfo
    elif mode == 'global':
        return sub_correlative_feas, metainfo
    else:
        return


def batch_coenrichment(fea, graph, safe_scores, n_iter=5000, p_value=0.05, _pre_cal_enriched=None):
    """
    Batch find with given feature at all possible features found at safe_scores.
    If _pre_cal_enriched was given, n_iter and p_value will be useless. Or you should modify n_iter and p_value as the params you passed to ``SAFE`` algorithm.

    :param str/list fea: A single feature or a list of feature which is already applied SAFE algorithm.
    :param graph:
    :param dict safe_scores: A SAFE score output from ``SAFE_batch`` which must contain all values occur at fea.
    :param int n_iter: Permutation times used at ``SAFE_batch``.
    :param float p_value: The p-value to determine the enriched nodes.
    :param dict _pre_cal_enriched: A pre calculated enriched_centroid which comprised all necessary features will save time.
    :return:
    """
    global_correlative_feas = {}
    sub_correlative_feas = {}
    metainfo = {}
    print('building network...')

    if '__iter__' in dir(fea):
        fea_batch = list(fea)[::]
    elif type(fea) == str:
        fea_batch = [fea]
    else:
        raise SyntaxError

    for fea in fea_batch:
        if fea in safe_scores.keys():
            if not _pre_cal_enriched:
                min_p_value = 1.0 / (n_iter + 1.0)
                threshold = np.log10(p_value) / np.log10(min_p_value)
                enriched_centroid, enriched_nodes = get_enriched_nodes(pd.DataFrame.from_dict(safe_scores, orient='index'), threshold, graph, centroids=True)
            else:
                enriched_centroid = _pre_cal_enriched
            _global, _local, _meta = coenrichment_for_nodes(graph, enriched_centroid[fea], fea, enriched_centroid, mode='both')
            global_correlative_feas.update(_global)
            sub_correlative_feas.update(_local)
            metainfo.update(_meta)
    return global_correlative_feas, sub_correlative_feas, metainfo


def construct_correlative_metadata(fea, global_correlative_feas, sub_correlative_feas, metainfo, node_data, verbose=1):
    """
    Down-stream transformation of ``batch_coenrichment``.
    :param fea:
    :param global_correlative_feas:
    :param sub_correlative_feas:
    :param metainfo:
    :param node_data:
    :return:
    """
    if verbose:
        print('Transforming given correlative relationship into human-readable DataFrame......')
    # processing global correlative feas
    global_headers = ['other feature',
                      'fisher-exact test pvalue',
                      'ranksum in co-enriched nodes',
                      'ranksum in others nodes',
                      'coverage/%', ]

    sub_headers = ['n_comps',
                   'comps_size',
                   'other feature',
                   'Fisher test pvalue(co-enriched,enriched)',
                   'Fisher test pvalue(co-enriched,others)',
                   'coenriched-enriched pvalue',
                   'coenriched-others pvalue',
                   'enriched-others pvalue',
                   'coverage/%',
                   ]
    global_corr_df = pd.DataFrame(columns=global_headers)
    for o_f in list(global_correlative_feas.keys()):
        _1, f_p = global_correlative_feas[o_f]
        s1, s2, s3, s4 = metainfo[o_f]

        if o_f in node_data.columns:
            _data = node_data
        else:
            print('error feature %s' % o_f)
            return
        y1 = _data.loc[s1, o_f]
        y2 = _data.loc[set.union(s2, s3, s4), o_f]

        if fea in node_data.columns:
            _data = node_data
        else:
            print('error feature %s' % o_f)
            return
        _y1 = _data.loc[s1, fea]
        _y2 = _data.loc[set.union(s2, s3, s4), fea]
        ranksum_p1 = scs.ranksums(y1, y2)[1]
        ranksum_p2 = scs.ranksums(_y1, _y2)[1]
        if not len(s1) + len(s3):
            coverage = np.nan
        else:
            coverage = ((len(s1) + len(s2)) / (len(s1) + len(s3))) * 100

        global_corr_df = global_corr_df.append(pd.DataFrame([[o_f, f_p, ranksum_p1, ranksum_p2, coverage]], columns=global_headers))

    # processing subgraph correlative feas
    sub_corr_df = pd.DataFrame(columns=sub_headers)
    for n_c, size, o_f in sub_correlative_feas.keys():
        if o_f in node_data.columns:
            _data = node_data
        else:
            print('error feature %s' % o_f)
            return
        coenriched_nodes, enriched_nodes_o_f_enriched, nonenriched_nodes_o_f_enriched = metainfo[(n_c, size, o_f)]
        f_p1, f_p2 = sub_correlative_feas[(n_c, size, o_f)]
        y1 = _data.loc[coenriched_nodes, o_f]
        y2 = _data.loc[enriched_nodes_o_f_enriched, o_f]
        y3 = _data.loc[nonenriched_nodes_o_f_enriched, o_f]
        p1 = scs.ranksums(y1, y2)[1]
        p2 = scs.ranksums(y1, y3)[1]
        p3 = scs.ranksums(y2, y3)[1]
        coverage = len(coenriched_nodes) / len(set.union(coenriched_nodes, enriched_nodes_o_f_enriched, nonenriched_nodes_o_f_enriched)) * 100
        sub_corr_df = sub_corr_df.append(pd.DataFrame([['comps%s' % n_c, size, o_f, f_p1, f_p2, p1, p2, p3, coverage]], columns=sub_headers))

    return global_corr_df, sub_corr_df


def pairwise_coenrichment(graph, safe_scores, n_iter=5000, p_value=0.05, _pre_cal_enriched=None, verbose=1):
    """
    Pair-wise calculation for co-enrichment of each feature found at safe_scores.
    If _pre_cal_enriched was given, n_iter and p_value is not useless.
    Or you should modify n_iter and p_value to fit the params you passed to ``SAFE`` algorithm.

    :param graph:
    :param dict safe_scores: A SAFE score output from ``SAFE_batch`` which must contain all values occur at fea.
    :param int n_iter: Permutation times used at ``SAFE_batch``.
    :param float p_value: The p-value to determine the enriched nodes.
    :param dict _pre_cal_enriched: A pre calculated enriched_centroid which comprised all necessary features will save time.
    :param verbose:
    :return:
    """
    dist_matrix = pd.DataFrame(data=np.nan, index=safe_scores.keys(), columns=safe_scores.keys())
    if verbose:
        print('building network...')
        iter_obj = tqdm(safe_scores.keys())
    else:
        iter_obj = safe_scores.keys()

    if not _pre_cal_enriched:
        min_p_value = 1.0 / (n_iter + 1.0)
        SAFE_pvalue = np.log10(p_value) / np.log10(min_p_value)
        enriched_centroid, enriched_nodes = get_enriched_nodes(pd.DataFrame.from_dict(safe_scores, orient='index'), SAFE_pvalue, graph, centroids=True)
    else:
        enriched_centroid = _pre_cal_enriched

    for fea in iter_obj:
        _global, _meta = coenrichment_for_nodes(graph, enriched_centroid[fea], fea, enriched_centroid, mode='global', _filter=False)
        # _filter to fetch raw fisher-exact test result without any cut-off values.
        for o_f in safe_scores.keys():
            if fea != o_f:
                s1, s2, s3, s4 = _meta[o_f]
                oddsratio, pvalue = _global[o_f]
                if is_enriched(s1, s2, s3, s4):
                    dist_matrix.loc[fea, o_f] = pvalue
                else:
                    dist_matrix.loc[fea, o_f] = 1
        dist_matrix.loc[fea, fea] = 0
    return dist_matrix
