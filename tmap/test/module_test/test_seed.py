from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps
from tmap.test import load_data

# load taxa abundance data, sample metadata and precomputed distance matrix
X = load_data.FGFP_genus_profile()
metadata = load_data.FGFP_metadata_ready()
dm = squareform(pdist(X, metric='braycurtis'))
############################################################
tm = mapper.Mapper(verbose=1)
metric = Metric(metric="precomputed")
lens = [Filter.PCOA(components=[0, 1], metric=metric)]  # for quick
projected_X = tm.filter(dm, lens=lens)
eps = optimize_dbscan_eps(X, threshold=95)
clusterer = DBSCAN(eps=eps, min_samples=3)
cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=50, overlap=0.75)
graph = tm.map(data=X, cover=cover, clusterer=clusterer)
node_data = graph.transform_sn(X)

from tmap.netx.SAFE import SAFE_batch, get_SAFE_summary

n_iter = 5000
safe_scores = SAFE_batch(graph, metadata=metadata, n_iter=n_iter, nr_threshold=0.5, _mode='both', random_seed=100)
enriched_SAFE_metadata, declined_SAFE_metadata = safe_scores['enrich'], safe_scores['decline']
safe_summary_metadata = get_SAFE_summary(graph=graph, metadata=metadata, safe_scores=enriched_SAFE_metadata,
                                         n_iter=n_iter, p_value=0.05)
############################################################
safe_scores = SAFE_batch(graph, metadata=metadata, n_iter=n_iter, nr_threshold=0.5, _mode='both', random_seed=500)
enriched_SAFE_metadata = safe_scores['enrich']
safe_summary_metadata2 = get_SAFE_summary(graph=graph, metadata=metadata, safe_scores=enriched_SAFE_metadata,
                                          n_iter=n_iter, p_value=0.05)
############################################################
n_iter = 5000
safe_scores = SAFE_batch(graph, metadata=metadata, n_iter=n_iter, nr_threshold=0.5, _mode='both', random_seed=100)
enriched_SAFE_metadata = safe_scores['enrich']
safe_summary_metadata3 = get_SAFE_summary(graph=graph, metadata=metadata, safe_scores=enriched_SAFE_metadata,
                                          n_iter=n_iter, p_value=0.05)

assert (safe_summary_metadata == safe_summary_metadata3).all().all()
assert (safe_summary_metadata != safe_summary_metadata2).any().any()
