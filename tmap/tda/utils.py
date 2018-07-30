from sklearn.neighbors import *
import numpy as np
import pandas as pd
import csv,os

def optimize_dbscan_eps(data, threshold=90):
    # using metric='minkowski', p=2 (that is, a euclidean metric)
    tree = KDTree(data, leaf_size=30, metric='minkowski', p=2)
    # the first nearest neighbor is itself, set k=2 to get the second returned
    dist, ind = tree.query(data, k=2)
    # to have a percentage of the 'threshold' of points to have their nearest-neighbor covered
    eps = np.percentile(dist[:, 1], threshold)
    return eps

def construct_node_data(graph,data,feature):
    nodes = graph['nodes']
    node_data = {k: data.iloc[v, data.columns.get_loc(feature)].mean() for k, v in nodes.items()}
    return node_data

def cover_ratio(graph,data):
    nodes = graph['nodes']
    all_samples_in_nodes = [_ for vals in nodes.values() for _ in vals]
    n_all_sampels = data.shape[0]
    n_in_nodes = len(set(all_samples_in_nodes))
    return n_in_nodes/float(n_all_sampels) *100

def safe_scores_IO(safe_scores,filepath=None,mode='w'):
    if mode == 'w':
        if not isinstance(safe_scores,pd.DataFrame):
            safe_scores = pd.DataFrame.from_dict(safe_scores,orient='index')
            safe_scores = safe_scores.T
        else:
            safe_scores = safe_scores
        safe_scores.to_csv(filepath,index=True)
    elif mode == 'rd':
        safe_scores = pd.read_csv(safe_scores,index_col=0)
        safe_scores = safe_scores.to_dict()
        return safe_scores
    elif mode == 'r':
        safe_scores = pd.read_csv(safe_scores,index_col=0)
        return safe_scores

def output_graph(graph,filepath,sep='\t'):
    """
    ouput graph as a file with sep [default=TAB]
    :param graph: Graph output from tda.mapper.map
    :param filepath:
    :param sep:
    """
    edges = graph['edges']
    with open(os.path.realpath(filepath),'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=sep)
        spamwriter.writerow(['Source', 'Target'])
        for source,target in edges:
            spamwriter.writerow([source,target])

def output_Node_data(graph,filepath,data,features = None,sep='\t',target_by='sample'):
    """
    output Node data with provided data.
    :param graph:
    :param filepath:
    :param data: pandas.Dataframe or np.ndarray with [n_samples,n_features] or [n_nodes,n_features]
    :param features: Array of features name
    :param sep:
    :param target_by: target type of "sample" or "node"
    :return:
    """
    if target_by not in ['sample','node']:
        exit("target_by should is one of ['sample','node']")
    nodes = graph['nodes']
    node_keys = graph['node_keys']
    if 'columns' in dir(data) and features is None:
        features = list(data.columns)
    elif 'columns' not in dir(data) and features is None:
        features = list(range(data.shape[1]))
    else:
        features = list(features)

    if type(data) != np.ndarray:
        data = np.array(data)

    if target_by == 'sample':
        data = np.array([np.mean(data[nodes[_]],axis=0) for _ in node_keys])
    else:
        pass

    with open(os.path.realpath(filepath),'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=sep)
        spamwriter.writerow(['NodeID'] + features)
        for idx,v in enumerate(node_keys):
            spamwriter.writerow([str(v)] + [str(_) for _ in data[idx,:]])

def output_Edge_data(graph,filepath,sep='\t'):
    """
    ouput edge data with sep [default=TAB]
    Mainly for netx.coenrich output
    :param graph: graph output by netx.co-enrich
    :param filepath:
    :param sep:
    :return:
    """
    if isinstance(graph,dict):
        if "edge_weights" in graph.keys() and "edges" in graph.keys():
            edges = graph["edges"]
            edge_weights = graph["edge_weights"]
            with open(os.path.realpath(filepath), 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=sep)
                spamwriter.writerow(["Edge name","coenrich_score"])
                for node1,node2 in edges:
                    spamwriter.writerow(["%s (interacts with) %s" % (node1,node2),
                                         edge_weights[(node1,node2)]])
        else:
            print("Missing key 'edge_weights' or 'edges' in graph")
    else:
        print("graph should be a dictionary")