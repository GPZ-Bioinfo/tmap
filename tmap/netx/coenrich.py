import itertools
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

def coenrich(graph, safe_scores):
    """
    Giving graph and safe_scores calculated by ``SAFE_batch``


    :param graph:
    :param safe_scores:
    :return:
    """
    nodes = graph['nodes']
    overall_coenrich = {}
    n_feas = len(safe_scores.keys())
    # iterative all fea1,fea2 without repeat.
    for fea1, fea2 in tqdm(itertools.combinations(safe_scores.keys(), 2), total=(n_feas ** 2 - n_feas) / 2):
        if sum(list(safe_scores[fea1].values())) != 0 or sum(list(safe_scores[fea2].values())) != 0:
            _fea1_vals = [safe_scores[fea1][_] for _ in nodes]
            _fea2_vals = [safe_scores[fea2][_] for _ in nodes]

            fea1_vals = [_fea1_vals[idx] for idx in range(len(nodes))]
            fea2_vals = [_fea2_vals[idx] for idx in range(len(nodes))]
            p_test = pearsonr(fea1_vals,
                              fea2_vals)
            overall_coenrich[(fea1, fea2)] = p_test

    graph_coenrich = {}
    graph_coenrich['edges'] = [k for k, v in overall_coenrich.items()]
    graph_coenrich['edge_coeffient'] = dict([(k, v[0]) for k, v in overall_coenrich.items()])
    graph_coenrich['edge_weights'] = dict([(k, v[1]) for k, v in overall_coenrich.items()])
    graph_coenrich['edge_adj_weights(fdr)'] = dict([(k, v) for k, v in zip(overall_coenrich.keys(),
                                                                           fdrcorrection([_[1] for _ in overall_coenrich.values()])[1])])
    return graph_coenrich
