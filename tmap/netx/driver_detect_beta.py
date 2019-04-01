####
# test pipelines which includes SHAP and xgboost.
# incoming pipeline which doesn't implement yet.
# lth 2018-12-10

# failed at 20190328
####

import itertools
import os
import pickle
import time

import numpy as np
import pandas as pd
import scipy
import shap
import xgboost as xgb
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score, auc, roc_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps

global_verbose = 1


def generate_graph(input_data, dis, _eu_dm=None, eps_threshold=95, overlap_params=0.75, min_samples=3, resolution_params="auto", filter_=Filter.PCOA):
    tm = mapper.Mapper(verbose=1)
    # TDA Step2. Projection
    t1 = time.time()
    metric = Metric(metric="precomputed")
    lens = [filter_(components=[0, 1], metric=metric, random_state=100)]
    projected_X = tm.filter(dis, lens=lens)
    if global_verbose:
        print("projection takes: ", time.time() - t1)
    ###
    t1 = time.time()
    eps = optimize_dbscan_eps(input_data, threshold=eps_threshold, dm=_eu_dm)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    r = resolution_params
    cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=r, overlap=overlap_params)
    graph = tm.map(data=input_data, cover=cover, clusterer=clusterer)
    if global_verbose:
        print(graph.info())
        print("graph time: ", time.time() - t1)

    graph_name = "{eps}_{overlap}_{r}_{filter}.graph".format(eps=eps_threshold, overlap=overlap_params, r=r, filter=lens[0].__class__.__name__)
    return graph, graph_name, projected_X


def read_graph(path):
    graph = pickle.load(open(path, 'rb'))
    return graph


def dump_graph(graph, path):
    pickle.dump(graph, open(path, "wb"))


def generate_XY(graph, input_data, center=False, weighted=True, beta=1):
    # If we consider the params passed to graph, we could make a more robust or faith X.
    t1 = time.time()

    def DiffusionKernel(AdjMatrix, beta):
        # 1.Computes Degree matrix  - diagonal matrix with diagonal entries = raw sums of adjacency matrix
        DegreeMatrix = np.diag(np.sum(AdjMatrix, axis=1))
        # 2. Computes negative Laplacian H = AdjMatrix - DegreeMatrix
        H = np.subtract(AdjMatrix, DegreeMatrix)
        # 3. Computes matrix exponential: exp(beta*H)
        K = scipy.linalg.expm(beta * H)
        return K

    node_data = graph.transform_sn(input_data)
    if "_raw_nodes" in graph["params"]:
        raw_nodes = graph['params']['_raw_nodes']
        node_ids = list(graph["nodes"].keys())
        adj_matrix = pd.DataFrame(data=np.nan, index=node_ids, columns=node_ids)
        for k1, k2 in itertools.combinations(node_ids, 2):
            if np.any(raw_nodes[k1] & raw_nodes[k2]):
                adj_matrix.loc[k1, k2] = 1
                adj_matrix.loc[k2, k1] = 1
        mask_array = (adj_matrix == 1)
    else:
        rng = np.arange(node_data.shape[0])
        mask_array = rng[:, None] < rng

    # prepare X and y
    if weighted:
        y = DiffusionKernel(graph["adj_matrix"].fillna(0).values, beta)[mask_array]
    else:
        y = graph["adj_matrix"].fillna(0).values[mask_array]

    edge_idx = graph["adj_matrix"].fillna(0).values[mask_array] == 1
    edge_data = np.ndarray((len(y), input_data.shape[1]))
    count_ = 0

    if global_verbose:
        _iter = tqdm(input_data.columns)
    else:
        _iter = input_data.columns

    for feature in _iter:
        one_di_data = np.abs(node_data.loc[:, feature].values - node_data.loc[:, feature].values.reshape(-1, 1))
        edge_data[:, count_] = one_di_data[mask_array]
        count_ += 1
    fetures = node_data.columns[edge_data.sum(0) != 0]

    edge_data = edge_data[:, edge_data.sum(0) != 0]
    if center:
        X = np.divide(edge_data, edge_data.std(axis=0))
    else:
        X = edge_data
    if global_verbose: print("Preparing X and y: ", time.time() - t1)
    return X, y, fetures, edge_idx


############################################################
def learn_rules(X, y, weighted=True, params=None):
    t1 = time.time()
    data = xgb.DMatrix(X, label=y)
    defaul_params = {"seed": 123,
                     "max_depth": 5,
                     "silent": 1,
                     "tree_method": "auto",
                     }
    num_boost_round = params.get("round", 100) if params is not None else 100
    if params is None:
        params = defaul_params
    else:
        defaul_params.update(params)
        params = defaul_params
    if weighted:
        xgb_params = {"objective": "reg:logistic",
                      "booster": "gbtree",
                      "eval_metric": "rmse",

                      # "subsample":0.3,
                      # "scale_pos_weight":pos_weight,
                      }
        xgb_params.update(params)
    else:
        pos_weight = len(y[y == 0]) / len(y[y == 1])
        xgb_params = {"objective": "binary:logistic",
                      "booster": "gbtree",
                      "scale_pos_weight": pos_weight,
                      }
        xgb_params.update(params)
    bst = xgb.train(xgb_params, data,
                    evals=[(data, "self")],
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=5,
                    verbose_eval=False)
    if global_verbose: print("xgboost taking: ", time.time() - t1)
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X)
    return shap_values, bst


def record_log(bst, graph_name, X, y, path, type="genera"):
    if os.path.isfile(path):
        f1 = open(path, mode="a")

    else:
        f1 = open(path, mode="w")
        f1.write("dtype\teps\toverlap\tr\tfilter\tr^2 score\n")

    y_predict = bst.predict(xgb.DMatrix(X, label=y))
    r2 = r2_score(y, y_predict)
    eps, overlap, r, filter_name = graph_name.strip(".graph").split("_")
    f1.write("\t".join([str(_) for _ in [type, eps, overlap, r, filter_name, r2]]) + '\n')


def eval_perform(y_true, y_pred, type="all"):
    """
    data = []
    fpr, tpr, th = precision_recall_curve(y_true, y_pred)
    data.append(go.Scatter(x=fpr,y=tpr))
    plotly.offline.plot(data)
    :param y_true:
    :param y_pred:
    :param type:
    :return:
    """
    if type == "pr":
        result = average_precision_score(y_true, y_pred)
    elif type == "auc":
        fpr, tpr, th = roc_curve(y_true, y_pred)
        result = auc(fpr, tpr, reorder=True)
    # elif type == "f1":
    #     result = f1_score(y_true, y_pred)
    elif type == "all":
        result = {}
        for i in ["pr", "auc", ]:
            result[i] = eval_perform(y_true, y_pred, type=i)
    else:
        result = ''
    return result


def cal_contri(shap_values, features_name):
    """
    contr_s = cal_contri(shap_values)
    :param shap_values:
    :return:
    """
    pos_sum = np.apply_along_axis(lambda x: x[x > 0].sum(), 1, shap_values)
    neg_sum = np.apply_along_axis(lambda x: x[x < 0].sum(), 1, shap_values)

    t1 = np.apply_along_axis(lambda x: x / pos_sum, 0, shap_values)
    t2 = np.apply_along_axis(lambda x: x / neg_sum, 0, shap_values)
    result = np.where(t1 > 0, t1, t2) / 2.0

    contr_s = pd.DataFrame(result, columns=features_name)
    return contr_s.mean(0).to_dict()


def full_pipelines(input_data,
                   weighted=True,
                   metric="braycurtis",
                   eps_threshold=95,
                   overlap_params=0.75,
                   resolution_params="auto",
                   filter_=Filter.PCOA):
    if type(metric) == str:
        dis = squareform(pdist(input_data, metric))
    elif "shape" in dir(metric):
        dis = metric
    else:
        dis = squareform(pdist(input_data))

    graph, graph_name, projected_X = generate_graph(input_data,
                                                    dis,
                                                    eps_threshold=eps_threshold,
                                                    overlap_params=overlap_params,
                                                    resolution_params=resolution_params,
                                                    filter_=filter_)
    X, y, features, edge_idx = generate_XY(graph, input_data, center=True, weighted=weighted)
    shap_values, bst = learn_rules(X, y, weighted=weighted)
    return shap_values, bst, features


if __name__ == '__main__':
    path = "../test/test_data/FGFP_genus_data.csv"
    genus_table = pd.read_csv(path, sep=',', index_col=0)
    shap_values, bst, features = full_pipelines(genus_table, weighted=True, resolution_params=35)
    shap_df = pd.DataFrame(shap_values, columns=features)
