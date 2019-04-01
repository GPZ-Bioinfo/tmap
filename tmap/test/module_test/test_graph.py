from sklearn import datasets
from sklearn.cluster import DBSCAN

from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
import numpy as np
import pandas as pd

import sklearn
assert sklearn.__version__ == "0.20.1"
############################################################
# prepare graph
X, y = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=100)
X = pd.DataFrame(X,index=['t%s' % _ for _ in range(X.shape[0])])
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
print(np.__version__)

assert len(graph.node) == 183
assert graph.node[2]['size'] == 28
assert graph.get_sample_size(2) == 28
assert graph.get_sample_size(30) == 33

assert graph.cover_ratio() - 0.9954 <= 1e-6

assert set(graph.node2sample(2)) == {'t1592', 't4921', 't4109', 't3514', 't4193',
                                     't2988', 't3263', 't2410', 't3763', 't1438',
                                     't3267', 't2545', 't3434', 't4769', 't4100',
                                     't3315', 't3118', 't2421', 't3742', 't808',
                                     't2561', 't3506', 't2745', 't2272', 't1291',
                                     't1909', 't3963', 't1284'}

assert set(graph.node2sample([5, 0, 1])) == {'t3810', 't1755', 't3157', 't990', 't1015', 't96', 't2988', 't2237', 't4101', 't2335', 't2090', 't2518', 't3440', 't3308', 't1147', 't1938', 't2079', 't4998', 't3372', 't3851', 't55', 't4533', 't2690', 't2610', 't2805', 't2042', 't892', 't4342', 't3315', 't1980', 't1053', 't832', 't764', 't999', 't496', 't2290', 't1436', 't604', 't3559', 't3508', 't808', 't3506', 't2745', 't1715', 't4410', 't497', 't1909', 't4564', 't2514', 't4204', 't937', 't2627'}

assert set(graph.sample2nodes(2)) == set(graph.sample2nodes('t2')) == {34, 40}
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
assert graph.get_neighborhoods(nodeid=[5, 2, 3], nr_threshold=4) == {5: [5, 4, 13, 14, 15],
                                                                     2: [2, 1, 3, 10, 11, 12],
                                                                     3: [3, 2, 4, 11, 12, 13]}
assert graph.get_neighborhoods(nodeid=[23]) == {23: [23, 14, 15, 24]}

neighborhodds = graph.get_neighborhoods(nr_threshold=4)
node_data = graph.transform_sn(X)
assert graph.neighborhood_score(node_data).iloc[32, 1] - -0.49188710882588677 <= 1e-6
assert graph.neighborhood_score(node_data,
                                neighborhoods=neighborhodds,
                                mode='mean').iloc[32, 1] - -0.12297177720647169 <= 1e-6
assert graph.neighborhood_score(node_data,
                                neighborhoods=neighborhodds,
                                mode='weighted_sum').iloc[32, 1] - -15.206617549205522 <= 1e-6
assert graph.neighborhood_score(node_data,
                                neighborhoods=neighborhodds,
                                mode='weighted_mean').iloc[32, 1] - -3.8016543873013804 <= 1e-6

assert graph.cubes.shape == (400, 5000)
assert graph.adjmatrix.shape == (183, 183)

assert graph.samples_neighbors(5, nr_threshold=10) == ['t98', 't132', 't134', 't135', 't6', 't73',
                                                       't120', 't110', 't111','t156', 't88', 't122',
                                                       't123', 't124', 't61', 't62', 't95']

assert graph.samples_neighbors('t5', nr_threshold=10) == ['t98', 't132', 't134', 't135', 't6', 't73', 't120',
                                                          't110', 't111','t156', 't88', 't122', 't123', 't124',
                                                          't61', 't62', 't95']
assert graph.samples_neighbors('t4997', nr_threshold=10) == []  # dropped samples

assert set(graph.samples_neighbors([2, 3], nr_threshold=5)) == {'t71', 't108', 't90', 't114', 't30', 't6', 't129', 't121', 't157',
       't86', 't65', 't24', 't25', 't107', 't128', 't73', 't156', 't58',
       't134', 't118', 't61', 't78', 't117', 't37', 't122', 't98', 't46',
       't72', 't82', 't85', 't64', 't63', 't19', 't77', 't16', 't158',
       't88', 't59', 't167'}

assert graph.get_shared_samples(5,4) == {'t2805', 't4101', 't604'}
assert graph.get_shared_samples(5,3) is None


assert graph.get_dropped_samples() == ['t4997', 't3848', 't1675', 't3470', 't3472', 't2461', 't4510',
       't2846', 't798', 't3489', 't2211', 't4389', 't298', 't826',
       't2241', 't4929', 't1606', 't1225', 't1486', 't719', 't343',
       't3163', 't4316']

assert graph.is_samples_shared('t1') == False    # case 1: no shared samples
assert graph.is_samples_shared('t4389') == False # case 2: dropped samples
assert graph.is_samples_shared('t4945') == True  # case 3: shared samples

assert graph.is_samples_dropped('t4389') == True
assert graph.is_samples_dropped('t1') == False

assert graph.data.shape[0] != len(graph.nodes)
