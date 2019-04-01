from __future__ import print_function
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.plot import Color
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps
from tmap.test import load_data
from scipy.spatial.distance import pdist,squareform

# load taxa abundance data, sample metadata and precomputed distance matrix
X = load_data.Daily_genus_profile("saliva")
metadata = load_data.Daily_metadata_ready()
dm = squareform(pdist(X,metric="braycurtis"))
metadata = metadata.loc[X.index,:]

# TDA Step1. initiate a Mapper
tm = mapper.Mapper(verbose=1)

# TDA Step2. Projection
metric = Metric(metric="precomputed")
lens = [Filter.MDS(components=[0, 1], metric=metric, random_state=100)]
projected_X = tm.filter(dm, lens=lens)

# Step4. Covering, clustering & mapping
eps = optimize_dbscan_eps(X, threshold=99)
clusterer = DBSCAN(eps=eps, min_samples=3)
cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=35, overlap=0.9)
graph = tm.map(data=X, cover=cover, clusterer=clusterer)
print(graph.info())


target_feature = 'COLLECTION_DAY'

color = Color(target=metadata.loc[:, target_feature], dtype="numerical", target_by="sample")
graph.show(color=color,
           fig_size=(10, 10),
           node_size=15,
           strength=0.03,notshow=True)


target_feature = 'Bacteroides'
color = Color(target=X.loc[:, target_feature], dtype="numerical", target_by="sample")
graph.show(color=color,
           fig_size=(10, 10),
           node_size=15,
           strength=0.03,notshow=True)

