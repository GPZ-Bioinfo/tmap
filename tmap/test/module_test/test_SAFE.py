from __future__ import print_function
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from tmap.netx.SAFE import SAFE_batch, get_SAFE_summary
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.plot import show, Color
from tmap.tda.utils import optimize_dbscan_eps
from tmap.test import load_data
from scipy.spatial.distance import squareform,pdist
# load taxa abundance data, sample metadata and precomputed distance matrix
X = load_data.FGFP_genus_profile()
metadata = load_data.FGFP_metadata_ready()
dm = squareform(pdist(X,metric='braycurtis'))

# TDA Step1. initiate a Mapper
tm = mapper.Mapper(verbose=1)

# TDA Step2. Projection
metric = Metric(metric="precomputed")
lens = [Filter.UMAP(components=[0, 1], metric=metric, random_state=100)]
projected_X = tm.filter(dm, lens=lens)

# Step4. Covering, clustering & mapping
eps = optimize_dbscan_eps(X, threshold=95)
clusterer = DBSCAN(eps=eps, min_samples=3)
cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=50, overlap=0.75)
graph = tm.map(data=X, cover=cover, clusterer=clusterer)

#prepare graph
############################################################
