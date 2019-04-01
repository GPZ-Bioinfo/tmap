# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
from pandas.api.types import is_categorical_dtype
from sklearn.neighbors import *
from sklearn.preprocessing import maxabs_scale


def optimize_dbscan_eps(data, threshold=90, dm=None):
    if dm is not None:
        tmp = dm.where(dm != 0, np.inf)
        eps = np.percentile(np.min(tmp, axis=0), threshold)
        return eps
    # using metric='minkowski', p=2 (that is, a euclidean metric)
    tree = KDTree(data, leaf_size=30, metric='minkowski', p=2)
    # the first nearest neighbor is itself, set k=2 to get the second returned
    dist, ind = tree.query(data, k=2)
    # to have a percentage of the 'threshold' of points to have their nearest-neighbor covered
    eps = np.percentile(dist[:, 1], threshold)
    return eps


#
# def optimal_r(X, projected_X, clusterer, mid, overlap, step=1):
#     def get_y(r):
#         from tmap.tda import mapper
#         from tmap.tda.cover import Cover
#         tm = mapper.Mapper(verbose=0)
#         cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=r, overlap=overlap)
#         graph = tm.map(data=X, cover=cover, clusterer=clusterer)
#         if "adj_matrix" not in graph.keys():
#             return np.inf
#         return abs(scs.skew(graph["adj_matrix"].count()))
#
#     mid_y = get_y(mid)
#     mid_y_r = get_y(mid + 1)
#     mid_y_l = get_y(mid - 1)
#     while 1:
#         min_r = sorted(zip([mid_y_l, mid_y, mid_y_r], [mid - 1, mid, mid + 1]))[0][1]
#         if min_r == mid - step:
#             mid -= step
#             mid_y, mid_y_r = mid_y_l, mid_y
#             mid_y_l = get_y(mid)
#         elif min_r == mid + step:
#             mid += step
#             mid_y, mid_y_l = mid_y_r, mid_y
#             mid_y_r = get_y(mid)
#         else:
#             break
#     print("suitable resolution is ", mid)
#     return mid


def unify_data(data):
    if 'iloc' in dir(data):
        # pd.DataFrame
        return data
    elif type(data) == list or isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    elif type(data) == dict:
        return pd.DataFrame.from_dict(data, orient='index')
    elif data is None:
        return
    else:
        print('Unkown data type. %s' % str(type(data)))
        return data


def transform2node_data(graph, data, mode='mean'):
    map_fun = {'sum': np.sum,
               'weighted_sum': np.sum,
               'weighted_mean': np.mean,
               "mean": np.mean}
    if mode not in ["sum", "mean", "weighted_sum", "weighted_mean"]:
        raise SyntaxError('Wrong provided parameters.')
    else:
        aggregated_fun = map_fun[mode]

    nodes = graph.nodes
    data = unify_data(data)
    dv = data.values
    if data is not None:
        node_data = {nid: aggregated_fun(dv[attr['sample'], :], 0) for nid, attr in nodes.items()}
        node_data = pd.DataFrame.from_dict(node_data, orient='index', columns=data.columns)
        return node_data


def transform2sample_data(graph, data):
    nodes = graph.nodes
    rawdata = unify_data(data)
    datas = []
    if rawdata is not None:
        for nid, attr in nodes.items():
            cache = [rawdata.loc[[nid], :]] * len(attr['sample'])
            cache = pd.concat(cache)
            cache.index = list(attr['sample'])
            datas.append(cache)
        sample_data = pd.concat(datas, axis=0)
        # todo: average the same index id row. result is larger than the number of origin sample
        return sample_data


def verify_metadata(graph, meta_data, by='node'):
    """
    DO:
      1. transform metadata into ``pd.DataFrame``
      2. transpose the matrix if necessary
      3. remove categorical columns in case raise error

    :param tmap.tda.Graph.Graph graph:
    :param meta_data:
    :return:
    """
    meta_data = unify_data(meta_data)
    if meta_data.shape[0] != graph.rawX.shape[0] and meta_data.shape[1] == graph.rawX.shape[0]:
        print('It may be a transposited matrix. it should be samples X OTU/features. So we will transposited it for you.')
        meta_data = meta_data.T
    elif meta_data.shape[0] != graph.rawX.shape[0] and meta_data.shape[1] != graph.rawX.shape[0]:
        raise SyntaxError("Wrong metadata provided. row should be samples(even the column isn't sample)...(Wrong number detected)")

    all_cat = np.array([is_categorical_dtype(meta_data.loc[:, col]) for col in meta_data])
    if any(all_cat):
        print("Detecting categorical column, it will automatically remove it and go on the anaylsis.")
        meta_data = meta_data.loc[:, ~all_cat]
        if meta_data.shape[1] == 0:
            exit('no columns remaining... (So sad....>.<)')

    if by == 'node':
        meta_data = graph.transform_sn(meta_data, type='s2n')
    else:
        pass
        # don't do anything.
    return meta_data


def output_fig(fig, output, mode):
    if mode == 'html' or output.endswith('html'):
        plotly.offline.plot(fig, filename=output, auto_open=False)
    else:
        pio.write_image(fig, output, format=mode)


#
# def dump_graph(graph, path, method='pickle'):
#     # method must one of 'pickle' or 'json'.
#     if method == 'pickle':
#         pickle.dump(graph, open(path, "wb"))
#     elif method == 'json':
#         # currently it will raise error because json can't dump ndarry directly.
#         json.dump(graph, open(path, 'w'))
#     else:
#         print('Wrong method provided, currently acceptable method are [pickle|json].')
#
#
# def output_graph(graph, filepath, sep='\t'):
#     """
#     Export graph as a file with sep. The output file should be used with `Cytoscape <http://cytoscape.org/>`_ .
#
#     It should be noticed that it will overwrite the file you provided.
#
#     :param dict graph: Graph output from tda.mapper.map
#     :param str filepath:
#     :param str sep:
#     """
#     edges = graph['edges']
#     with open(os.path.realpath(filepath), 'w') as csvfile:
#         spamwriter = csv.writer(csvfile, delimiter=sep)
#         spamwriter.writerow(['Source', 'Target'])
#         for source, target in edges:
#             spamwriter.writerow([source, target])


#
# def output_Node_data(graph,filepath,data,features = None,sep='\t',target_by='sample'):
#     """
#     Export Node data with provided filepath. The output file should be used with `Cytoscape <http://cytoscape.org/>`_ .
#
#     It should be noticed that it will overwrite the file you provided.
#
#     :param dict graph:
#     :param str filepath:
#     :param np.ndarray/pandas.Dataframe data: with shape [n_samples,n_features] or [n_nodes,n_features]
#     :param list features: It could be None and it will use count number as feature names.
#     :param str sep:
#     :param str target_by: target type of "sample" or "node"
#     """
#     if target_by not in ['sample','node']:
#         exit("target_by should is one of ['sample','node']")
#     nodes = graph['nodes']
#     node_keys = graph['node_keys']
#     if 'columns' in dir(data) and features is None:
#         features = list(data.columns)
#     elif 'columns' not in dir(data) and features is None:
#         features = list(range(data.shape[1]))
#     else:
#         features = list(features)
#
#     if type(data) != np.ndarray:
#         data = np.array(data)
#
#     if target_by == 'sample':
#         data = np.array([np.mean(data[nodes[_]],axis=0) for _ in node_keys])
#     else:
#         pass
#
#     with open(os.path.realpath(filepath),'w') as csvfile:
#         spamwriter = csv.writer(csvfile, delimiter=sep)
#         spamwriter.writerow(['NodeID'] + features)
#         for idx,v in enumerate(node_keys):
#             spamwriter.writerow([str(v)] + [str(_) for _ in data[idx,:]])
def write_figure(fig, mode, **kwargs):
    if mode == 'file':
        plotly.offline.plot(fig, **kwargs)

    elif mode == 'web':
        plotly.offline.iplot(fig, **kwargs)
    elif mode == 'obj':
        return fig
    else:
        r = input("mode params must be one of 'file', 'web', 'obj'. \n 'file': output html file \n 'web': show in web browser. \n 'obj': return a dict object.")
        if r.lower() in ['file', 'web', 'obj']:
            write_figure(fig, mode=r)


def c_node_text(nodes, sample_names, target_v_raw):
    # values output from color.target. It need to apply mean function for a samples-length color.target.
    node_text = [str(n) +
                 # node id
                 "<Br>vals:%s<Br>" % '{}'.format(v) +
                 # node values output from color.target.
                 '<Br>'.join(list(sample_names[nodes[n]['sample']][:8]) + ['......']
                             if len(sample_names[nodes[n]['sample']]) >= 8  # too long will make the node doesn't hover anythings.
                             else sample_names[nodes[n]['sample']]) for n, v in
                 # samples name concated with line break.
                 zip(nodes,
                     target_v_raw)]
    return node_text


# accessory for envfit

def get_arrows(graph, safe_score, max_length=1, pvalue=0.05):
    min_p_value = 1.0 / (5000 + 1.0)
    threshold = np.log10(pvalue) / np.log10(min_p_value)

    node_pos = graph.nodePos

    safe_score_df = pd.DataFrame.from_dict(safe_score, orient='columns')
    safe_score_df = safe_score_df.where(safe_score_df >= threshold, other=0)
    norm_df = safe_score_df.apply(lambda x: maxabs_scale(x), axis=1, result_type='broadcast')

    x_cor = norm_df.apply(lambda x: x * node_pos.values[:, 0], axis=0)
    y_cor = norm_df.apply(lambda x: x * node_pos.values[:, 1], axis=0)

    x_cor = x_cor.mean(0)
    y_cor = y_cor.mean(0)
    arrow_df = pd.DataFrame([x_cor, y_cor], index=['x coordinate', 'y coordinate'], columns=safe_score_df.columns)
    all_fea_scale = maxabs_scale(safe_score_df.sum(0))
    # scale each arrow by the sum of safe score， maximun is 1 others are percentage not larger than 100%.
    scaled_ratio = max_length * all_fea_scale / arrow_df.apply(lambda x: np.sqrt(np.sum(x ** 2)), axis=0)
    # using max length to multipy by scale ratio and denote the original length.
    scaled_arrow_df = arrow_df * np.repeat(scaled_ratio.values.reshape(1, -1), axis=0, repeats=2).reshape(2, -1)

    return scaled_arrow_df
