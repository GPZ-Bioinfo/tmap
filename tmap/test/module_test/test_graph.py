from sklearn import datasets
from sklearn.cluster import DBSCAN

from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover

############################################################
# prepare graph
X, y = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=100)

# Step1. initiate a Mapper
tm = mapper.Mapper(verbose=1)

# Step2. Projection
lens = [Filter.PCA(components=[0, 1])]
projected_X = tm.filter(X, lens=lens)

# Step3. Covering, clustering & mapping
clusterer = DBSCAN(eps=0.1, min_samples=5)
cover = Cover(projected_data=projected_X, resolution=20, overlap=0.1)
graph = tm.map(data=X, cover=cover, clusterer=clusterer)
############################################################
# start test

assert len(graph.node) == 183
assert graph.node[2]['size'] == 28
assert graph.get_sample_size(2) == 28
assert graph.get_sample_size(30) == 33

assert graph.cover_ratio() - 0.9954 <= 1e-6

assert set(graph.node2sample(2)) == {2561, 1284, 4100, 1291, 4109, 1438, 3742, 4769, 808, 2988, 3118,
                                     3506, 3763, 1592, 2745, 3514, 4921, 3263, 3267, 2272, 4193, 2410,
                                     3434, 2545, 3315, 1909, 2421, 3963}
assert set(graph.node2sample([5, 0, 1])) == {2690, 764, 4101, 4998, 3851, 1938, 1436, 1053, 2335, 2079, 808,
                                             937, 2090, 2988, 3372, 2610, 1715, 3506, 4533, 3508, 55, 2745,
                                             4410, 1980, 2237, 832, 2627, 2514, 4564, 3157, 2518, 1755, 604,
                                             990, 96, 3810, 3559, 999, 3308, 4204, 496, 497, 2290, 3315,
                                             3440, 2805, 1909, 1015, 4342, 2042, 1147, 892}

assert set(graph.sample2nodes(2)) == {34, 40}
assert set(graph.sample2nodes([3372, 2610, 1715, 3506, 4533, 3508, 55, 2745, ])) == {0, 1, 2, 5, 9,
                                                                                     10, 11, 14}
assert graph.transform_sn(X).shape[0] == 183
assert graph.transform_sn(graph.transform_sn(X), type='n2s').shape[0] == 7075


graph.update_dist()
assert set(graph.all_spath[0][174]) == {0, 1, 2, 3, 4, 5, 15, 25, 33, 40, 46, 47, 57,
                                        67, 79, 90, 101, 114, 126, 134, 140, 146, 153, 152, 165, 175,
                                        174}
assert graph.all_length[0][174] == 26


graph.update_dist(weight='dist')
assert set(graph.all_spath[0][174]) == {0, 9, 8, 20, 31, 30, 37, 43, 42, 49, 59, 69, 82,
                                        93, 104, 117, 129, 137, 142, 143, 150, 157, 158, 159, 160, 161,
                                        173, 174}
assert graph.all_length[0][174] - 3.169179651599512 <= 1e-6


graph.update_dist()
assert graph.get_neighborhoods(nodeid=[5,2,3],nr_threshold=4) == {5: [5, 4, 13, 14, 15],
                                                                  2: [2, 1, 3, 10, 11, 12],
                                                                  3: [3, 2, 4, 11, 12, 13]}
assert graph.get_neighborhoods(nodeid=[23]) == {23: [23]}

neighborhodds = graph.get_neighborhoods(nr_threshold=4)
node_data = graph.transform_sn(X)
assert graph.neighborhood_score(node_data).iloc[32,1] - -0.11744862933577845 <= 1e-6
assert graph.neighborhood_score(node_data,
                                neighborhoods=neighborhodds,
                                mode='mean').iloc[32,1] - -0.12297177720647169 <= 1e-6
assert graph.neighborhood_score(node_data,
                                neighborhoods=neighborhodds,
                                mode='weighted_sum').iloc[32,1] - -15.206617549205522 <= 1e-6
assert graph.neighborhood_score(node_data,
                                neighborhoods=neighborhodds,
                                mode='weighted_mean').iloc[32,1] - -3.8016543873013804 <= 1e-6


assert graph.cubes.shape == (400,5000)
assert graph.adjmatrix.shape == (183,183)