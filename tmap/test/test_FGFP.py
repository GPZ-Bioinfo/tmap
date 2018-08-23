from __future__ import print_function
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from tmap.tda import mapper, filter
from tmap.tda.cover import Cover
from tmap.tda.plot import show, Color
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps,cover_ratio
from tmap.netx.SAFE import SAFE_batch, get_SAFE_summary
from tmap.test import load_data
import os

# load taxa abundance data, sample metadata and precomputed distance matrix
X = load_data.FGFP_genus_profile()
metadata = load_data.FGFP_metadata_ready()
dm = load_data.FGFP_BC_dist()

# TDA Step1. initiate a Mapper
tm = mapper.Mapper(verbose=1)

# TDA Step2. Projection
metric = Metric(metric="precomputed")
lens = [filter.MDS(components=[0, 1], metric=metric,random_state=100)]
projected_X = tm.filter(dm, lens=lens)

# Step4. Covering, clustering & mapping
eps = optimize_dbscan_eps(X, threshold=95,dm=dm)
clusterer = DBSCAN(eps=eps, min_samples=3)
cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=50, overlap=0.75)
graph = tm.map(data=X, cover=cover, clusterer=clusterer)
print('Graph covers %.2f percentage of samples.' % cover_ratio(graph,X))

## Step 6. SAFE test for every features.

# target_feature = 'Faecalibacterium'
# target_feature = 'Prevotella'
target_feature = 'Bacteroides'
n_iter = 1000
safe_scores = SAFE_batch(graph, meta_data=X, n_iter=n_iter, threshold=0.05)
target_safe_score = safe_scores[target_feature]

# target_safe_score = SAFE_single(graph, X.loc[:, target_feature], n_iter=1000, threshold=0.05)

## Step 7. Visualization

# colors by samples (target values in a list)
color = Color(target=X.loc[:, target_feature], dtype="numerical", target_by="sample")
show(data=X, graph=graph, color=color, fig_size=(10, 10), node_size=15, mode='spring', strength=0.08)

# colors by nodes (target values in a dictionary)
color = Color(target=target_safe_score, dtype="numerical", target_by="node")
show(data=X, graph=graph, color=color, fig_size=(10, 10), node_size=15, mode='spring', strength=0.08)

safe_summary = get_SAFE_summary(graph=graph, meta_data=X, safe_scores=safe_scores,
                                n_iter_value=n_iter, p_value=0.01)
safe_summary.to_csv('/home/liaoth/data2/project/TDA_paper/compare_/genus_SAFE_v2.csv', index=True)
safe_summary.to_csv(os.path.join(os.path.dirname(__file__),'../example/FGFP_SAFE_ranking_genus_1000.csv'), index=True)
