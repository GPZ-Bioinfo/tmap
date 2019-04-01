from __future__ import print_function

from matplotlib.pyplot import title
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.plot import Color
from tmap.tda.utils import optimize_dbscan_eps
from tmap.test import load_data

# load taxa abundance data, sample metadata and precomputed distance matrix
X = load_data.Daily_genus_profile("stool")
X = X.drop("Stool69", axis=0)
metadata = load_data.Daily_metadata_ready()
dm = squareform(pdist(X, metric="braycurtis"))
metadata = metadata.loc[X.index, :]

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
print(graph.info())

target_feature = 'COLLECTION_DAY'
color = Color(target=metadata.loc[:, target_feature],
              dtype="numerical",
              target_by="sample")
graph.show(color=color, fig_size=(10, 10), node_size=15, notshow=True)

target_feature = 'HOST_SUBJECT_ID'
color = Color(target=metadata.loc[:, target_feature],
              dtype="categorical",
              target_by="sample")
graph.show(color=color, fig_size=(10, 10), node_size=15, notshow=True)

color = Color(target=metadata.loc[:, target_feature],
              dtype="numerical",
              target_by="sample")
graph.show(color=color, fig_size=(10, 10), node_size=15, notshow=True)


def time_range(sample, start, end):
    target_vals = [1 if metadata.loc[_, "HOST_SUBJECT_ID"] == "2202:Donor%s" % sample and metadata.loc[_, "COLLECTION_DAY"] in list(range(start, end + 1)) else 0 for _ in X.index]
    color = Color(target=target_vals, dtype="numerical", target_by="sample")
    graph.show(color=color, fig_size=(10, 10), node_size=15,notshow=True)
    title("Subject %s at %s to %s" % (sample, start, end))


# Travel period
time_range("A", 70, 123)
# First diarrheal illness
time_range("A", 80, 85)
# Second diarrheal illness
time_range("A", 104, 113)

# Pre-travel period
time_range("A", 40, 69)
# Travel period
time_range("A", 70, 122)
# Post-travel period
time_range("A", 123, 153)

############################################################
from tmap.netx.SAFE import SAFE_batch

n_iter = 1000
safe_scores = SAFE_batch(graph,
                         X,
                         n_iter=n_iter,
                         nr_threshold=0.05,
                         _mode='both',
                         name='For genus table')
enriched_scores, declined_scores = safe_scores['enrich'], safe_scores['decline']
safe_scores = SAFE_batch(graph,
                         metadata.iloc[:, 4:],
                         n_iter=n_iter,
                         nr_threshold=0.05,
                         _mode='both',
                         name='For metadata')
enriched_meta_scores, declined_meta_scores = safe_scores['enrich'], safe_scores['decline']

from tmap.tda.plot import Color
graph.show(color=Color(X.loc[:,'Bacteroides'],target_by='node'),notshow=True)