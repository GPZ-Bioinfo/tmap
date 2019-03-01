from __future__ import print_function
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.plot import show, Color
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps,cover_ratio
from tmap.test import load_data

from matplotlib.pyplot import title
from scipy.spatial.distance import pdist,squareform

# load taxa abundance data, sample metadata and precomputed distance matrix
X = load_data.Daily_genus_profile("stool")
X = X.drop("Stool69",axis=0)
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
cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=50, overlap=0.85)
graph = tm.map(data=X, cover=cover, clusterer=clusterer)
print('Graph covers %.2f percentage of samples.' % cover_ratio(graph,X))

target_feature = 'COLLECTION_DAY'
color = Color(target=metadata.loc[:, target_feature], dtype="numerical", target_by="sample")
show(data=X, graph=graph, color=color, fig_size=(10, 10), node_size=15, mode='spring', strength=0.12)

target_feature = 'HOST_SUBJECT_ID'
color = Color(target=metadata.loc[:, target_feature], dtype="categorical", target_by="sample")
show(data=X, graph=graph, color=color, fig_size=(10, 10), node_size=15, mode='spring', strength=0.12)

color = Color(target=metadata.loc[:, target_feature], dtype="numerical", target_by="sample")
show(data=X, graph=graph, color=color, fig_size=(10, 10), node_size=15, mode='spring', strength=0.12)

def time_range(sample,start,end):
    target_vals = [1 if metadata.loc[_,"HOST_SUBJECT_ID"]=="2202:Donor%s" % sample and metadata.loc[_,"COLLECTION_DAY"] in list(range(start,end+1)) else 0 for _ in X.index]
    color = Color(target=target_vals, dtype="numerical", target_by="sample")
    show(data=X, graph=graph, color=color, fig_size=(10, 10), node_size=15, mode='spring', strength=0.12)
    title("Subject %s at %s to %s" % (sample,start,end))

# Travel period
time_range("A",70,123)
# First diarrheal illness
time_range("A",80,85)
# Second diarrheal illness
time_range("A",104,113)

# Pre-travel period
time_range("A",40,69)
# Travel period
time_range("A",70,122)
# Post-travel period
time_range("A",123,153)

############################################################
# n_iter = 1000
# safe_scores = SAFE_batch(graph, meta_data=X, n_iter=n_iter, threshold=0.05)
# safe_scores_meta = SAFE_batch(graph, meta_data=metadata.iloc[:,4:], n_iter=n_iter, threshold=0.05)
#
# safe_scores.update(safe_scores_meta)
# co_vals = coenrich(graph,safe_scores)
#
# ############################################################
# meta_cols = list(metadata.iloc[:,4:].columns)
# genera = list(X.columns)
#
# co_meta2genus = [_ for _ in co_vals["edges"] if (_[0] in meta_cols and _[1] in genera) or (_[1] in meta_cols and _[0] in genera)]
#
# co_meta2genus_s = sorted([(co_vals['edge_adj_weights(fdr)'][_],_) for _ in co_meta2genus])