from __future__ import print_function

from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from tmap.netx.SAFE import SAFE_batch
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps
from tmap.test import load_data

# load taxa abundance data, sample metadata and precomputed distance matrix
X = load_data.FGFP_genus_profile()
metadata = load_data.FGFP_metadata_ready()
dm = squareform(pdist(X, metric='braycurtis'))

# TDA Step1. initiate a Mapper
tm = mapper.Mapper(verbose=1)

# TDA Step2. Projection
metric = Metric(metric="precomputed")
lens = [Filter.UMAP(components=[0, 1], metric=metric, random_state=100)]  # for quick
projected_X = tm.filter(dm, lens=lens)

# Step4. Covering, clustering & mapping
eps = optimize_dbscan_eps(X, threshold=95)
clusterer = DBSCAN(eps=eps, min_samples=3)
cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=50, overlap=0.75)
graph = tm.map(data=X, cover=cover, clusterer=clusterer)

# prepare graph
############################################################

safe_scores = SAFE_batch(graph, metadata, n_iter=50, _mode='both')
enriched_scores, declined_scores = safe_scores['enrich'],safe_scores['decline']
assert enriched_scores.shape == (952, 174)
assert declined_scores.shape == (952, 174)

enriched_scores = SAFE_batch(graph, metadata, n_iter=50, _mode='enrich')
assert enriched_scores.shape == (952, 174)

safe_scores = SAFE_batch(graph, metadata, n_iter=50, shuffle_by='sample', _mode='both')
enriched_scores, declined_scores = safe_scores['enrich'],safe_scores['decline']
assert enriched_scores.shape == (952, 174)
assert declined_scores.shape == (952, 174)

from tmap.netx.SAFE import get_significant_nodes

significant_centroids, significant_nodes = get_significant_nodes(graph,
                                                                 enriched_scores,
                                                                 r_neighbor=True)

from tmap.netx.SAFE import get_SAFE_summary

safe_summary = get_SAFE_summary(graph,
                                metadata,
                                enriched_scores)
