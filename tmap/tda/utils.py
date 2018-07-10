from sklearn.neighbors import *
import numpy as np


def optimize_dbscan_eps(data, threshold=90):
    # using metric='minkowski', p=2 (that is, a euclidean metric)
    tree = KDTree(data, leaf_size=30, metric='minkowski', p=2)
    # the first nearest neighbor is itself, set k=2 to get the second returned
    dist, ind = tree.query(data, k=2)
    # to have a percentage of the 'threshold' of points to have their nearest-neighbor covered
    eps = np.percentile(dist[:, 1], threshold)
    return eps
