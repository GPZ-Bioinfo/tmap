############################################################
# test color
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.cluster import DBSCAN

from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.plot import Color

assert sklearn.__version__ == "0.20.1"
############################################################
# prepare graph
X, y = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=100)
X = pd.DataFrame(X, index=['t%s' % _ for _ in range(X.shape[0])])

metadata = pd.DataFrame(np.zeros((X.shape[0], 1)),
                        index=X.index,
                        columns=['num'])
metadata.loc[:, 'cat'] = [v for alpha in 'abcde' for v in np.repeat(alpha, 1000)]
metadata.loc[:, 'num'] = X.iloc[:, 1]

# Step1. initiate a Mapper
tm = mapper.Mapper(verbose=1)

# Step2. Projection
lens = [Filter.PCA(components=[0, 1])]
projected_X = tm.filter(X, lens=lens)

# Step3. Covering, clustering & mapping
clusterer = DBSCAN(eps=0.1, min_samples=5)
cover = Cover(projected_data=projected_X,
              resolution=20,
              overlap=0.1)
graph = tm.map(data=X,
               cover=cover,
               clusterer=clusterer)
############################################################
metadata = pd.get_dummies(metadata)
metadata.loc[:, 'circle'] = y
node_metadata = graph.transform_sn(metadata, type='s2n')
############################################################
color1 = Color(target=node_metadata.iloc[:, 0],
               dtype='numerical',
               target_by='sample')
# color1.get_colors(graph.nodes)


color2 = Color(target=node_metadata.iloc[:, 0],
               dtype='numerical',
               target_by='node')
color3 = Color(target=metadata.iloc[:, 0],
               dtype='numerical',
               target_by='sample')
assert np.all(color3.get_colors(graph.nodes)[1][1] == color2.get_colors(graph.nodes)[1][1])

color4 = Color(target=node_metadata.iloc[:, 1],
               dtype='categorical',
               target_by='node')  # wrong example, it should not use it as this way
color5 = Color(target=metadata.iloc[:, 1],
               dtype='categorical',
               target_by='sample')
assert np.all(color4.get_colors(graph.nodes)[1][1] != color5.get_colors(graph.nodes)[1][1])

assert set(color5.get_colors(graph.nodes)[1][1]) == {'#0000bf'}
assert color5.get_sample_colors()[1] == {1: '#bf0000', 0: '#0000bf'}
assert color5.get_sample_colors(cmap={1: 'blue',
                                      0: 'yellow'})[1] == {1: 'blue',
                                                           0: 'yellow'}
assert set(color5.get_colors(graph.nodes, cmap={1: 'blue',
                                                0: 'yellow'})[1][0][:, 0]) == {0}

############################################################
# %matplotlib
from tmap.netx.SAFE import SAFE_batch

safe_scores = SAFE_batch(graph,
                         metadata,
                         n_iter=500,
                         _mode='both')
enriched_scores, declined_scores = safe_scores['enrich'], safe_scores['decline']
color = Color(enriched_scores.loc[:,
              'circle'], target_by='node', dtype='numerical')
color6 = Color(metadata.loc[:, 'circle'], target_by='sample', dtype='numerical')
color7 = Color(metadata.loc[:, 'circle'], target_by='sample', dtype='categorical')
from tmap.tda.plot import show

show(graph,notshow=True)
show(graph, mode=None,notshow=True)
show(graph, color=color6,notshow=True)
show(graph, color=color7,notshow=True)
graph.show_samples(['t1', 't688', 't3000'],notshow=True)
graph.show(notshow=True)
from tmap.tda.plot import vis_progressX

vis_progressX(graph,
              color=color6,
              _color_SAFE=color,
              mode='file', auto_open=False)
vis_progressX(graph,
              color=color7,
              _color_SAFE=color,
              mode='file', auto_open=False)

vis_progressX(graph, color=color6, simple=True,auto_open=False)
