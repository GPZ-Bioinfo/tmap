from __future__ import print_function
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from tmap.netx.SAFE import SAFE_batch, get_SAFE_summary
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.plot import Color
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
print(graph.info())

## Step 6. SAFE test for every features.

# target_feature = 'Faecalibacterium'
# target_feature = 'Prevotella'
target_feature = 'Bacteroides'
n_iter = 1000
enriched_scores = SAFE_batch(graph,
                         metadata=X,
                         n_iter=n_iter,
                         _mode = 'enrich')
target_safe_score = enriched_scores.loc[:,target_feature]

## Step 7. Visualization

# colors by samples (target values in a list)
color = Color(target=X.loc[:, target_feature], dtype="numerical", target_by="sample")
graph.show(color=color,
           fig_size=(10, 10),
           node_size=15,

           notshow=True)

# colors by nodes (target values in a dictionary)
color = Color(target=target_safe_score, dtype="numerical", target_by="node")
graph.show(color=color,
           fig_size=(10, 10),
           node_size=15,

           notshow=True)

safe_summary = get_SAFE_summary(graph=graph,
                                metadata=X,
                                safe_scores=enriched_scores,
                                n_iter=n_iter,
                                p_value=0.01)