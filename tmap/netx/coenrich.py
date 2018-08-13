import itertools
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

def coenrich(graph, safe_scores):
    """
    Giving graph and safe_scores calculated by ``SAFE_batch``

    Currently, we using itertools.combinations for iterating all possibles combinations of each features and perform pearson test with corresponding SAFE scores. Then, it will perform multiple test correction with fdr_correction.

    Finally, it will output a dict with different keys.

    * associated_pairs: tuple of associated features.
    * association_coeffient: coeffient produced by pearson test
    * association_p_values: p-values produced by pearson test
    * association_p_values(fdr): p-values after multiple test correction

    SAFE score is a metric trying to capture the variation of corresponding feature and so it will result much 0 values for among the network.

    For pearson test, two array of values with lots of zeros will result some pseudo association. It should be careful it there are negative association with two features.
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
    graph_coenrich['associated_pairs'] = [k for k, v in overall_coenrich.items()]
    graph_coenrich['association_coeffient'] = dict([(k, v[0]) for k, v in overall_coenrich.items()])
    graph_coenrich['association_p_values'] = dict([(k, v[1]) for k, v in overall_coenrich.items()])
    graph_coenrich['association_p_values(fdr)'] = dict([(k, v) for k, v in zip(overall_coenrich.keys(),
                                                                           fdrcorrection([_[1] for _ in overall_coenrich.values()])[1])])
    return graph_coenrich
