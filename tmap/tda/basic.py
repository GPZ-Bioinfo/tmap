import time

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps, cover_ratio, optimal_r
from tmap.api.general import logger


def generate_graph(input_data, dis, _eu_dm=None, eps_threshold=95, overlap_params=0.75, min_samples=3, resolution_params="auto", filter_=Filter.PCOA, verbose=1):
    tm = mapper.Mapper(verbose=1)
    # TDA Step2. Projection
    t1 = time.time()
    metric = Metric(metric="precomputed")
    lens = [filter_(components=[0, 1], metric=metric, random_state=100)]
    projected_X = tm.filter(dis, lens=lens)
    logger("projection takes: ", time.time() - t1, verbose=verbose)
    ###
    t1 = time.time()
    eps = optimize_dbscan_eps(input_data, threshold=eps_threshold, dm=_eu_dm)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    if resolution_params == "auto":
        r = optimal_r(input_data, projected_X, clusterer, 40, overlap_params)
    else:
        r = resolution_params
    cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=r, overlap=overlap_params)
    graph = tm.map(data=input_data, cover=cover, clusterer=clusterer)
    logger('Graph covers %.2f percentage of samples.' % cover_ratio(graph, input_data),verbose=verbose)
    logger("graph time: ", time.time() - t1,verbose=verbose)

    graph_name = "{eps}_{overlap}_{r}_{filter}.graph".format(eps=eps_threshold, overlap=overlap_params, r=r, filter=lens[0].__class__.__name__)
    return graph, graph_name, projected_X
