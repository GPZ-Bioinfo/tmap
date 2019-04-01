from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from sklearn.cluster import DBSCAN
from tmap.tda.plot import Color


digits = datasets.load_digits()
X = digits.data
y = digits.target

# Step1. initiate a Mapper
tm = mapper.Mapper(verbose=1)

# Step2. Projection
lens = [Filter.TSNE(components=[0, 1])]
projected_X = tm.filter(X, lens=lens)

# Step3. Covering, clustering & mapping
clusterer = DBSCAN(eps=3, min_samples=5)
cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=25, overlap=0.6)
graph = tm.map(data=MinMaxScaler().fit_transform(X), cover=cover, clusterer=clusterer)

# Step4. Visualization
color = Color(target=y, dtype="categorical")
graph.show(color=color, fig_size=(10, 10), node_size=10, notshow=True)

