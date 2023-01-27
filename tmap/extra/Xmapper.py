from tmap.tda.mapper import Mapper
from tmap.tda.Graph import Graph

import itertools

import numpy as np
from sklearn.base import ClusterMixin,BaseEstimator
from tqdm import tqdm
from scipy.cluster import hierarchy
from scipy.spatial import distance
import numpy as np

class hier_cluster(ClusterMixin, BaseEstimator):
    """
    Within the LocalClustering step, it require to perform hierarchical clustering and cutting with definable threshold.
    """
    def __init__(self, linkage_method='single',metric='euclidean',):
        # self.eps = eps
        # self.min_samples = min_samples
        self.metric = metric
        self.linkage_method = linkage_method
        # self.algorithm = algorithm
        # self.leaf_size = leaf_size
        # self.p = p
        # self.n_jobs = n_jobs
        pass
        
    def fit(self, X,):
        #dis_X = distance.pdist(X,metric=self.metric)
        # print('dis_X',X.shape)
        if X.shape[0]<=1:
            self.labels_ = np.array([])
            return 
        link = hierarchy.linkage(X,method=self.linkage_method,metric=self.metric)
        
        hist,bin_edges = np.histogram(hierarchy.cophenet(link),bins=10)
        # if there are not empty size bin
        # it should use the largest value as cut off instead of smallest value which will generate too many clusters
        if any(hist==0):
            low_bound_empty_hist = bin_edges[(hist==0).argmax()]
        else:
            low_bound_empty_hist = bin_edges[-1]
        cutree = hierarchy.cut_tree(link,height=low_bound_empty_hist)
        self.labels_ = cutree.reshape(-1)
    
    