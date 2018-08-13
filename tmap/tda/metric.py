# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance


# supported / allowed metrics
_METRIC_ALLOWED = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
                    "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto",
                    "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule",
                    "precomputed"]


class Metric(object):
    """
    ``metric + data -> distance matrix``

    Define a distance metric and transform data points into a distance matrix.

    :param str metric: `metric` specified a distance metric.
        For example:

        * cosine
        * euclidean
        * hamming
        * minkowski
        * precomputed:  for precomputed distance matrix.

    """

    def __init__(self, metric="euclidean"):
        if metric not in _METRIC_ALLOWED:
            raise Exception("The metric is not allowed: %s." % metric)
        self.name = metric

    def fit_transform(self, data):
        """
        Create and return a distance matrix based on the specified metric.

        :param np.ndarray/pd.DataFrame data: `data`: raw data or precomputed distance matrix.
        """

        if data is None:
            raise Exception("Data must not be None.")
        if type(data) is not np.ndarray:
            data = np.array(data)

        if self.name == "precomputed":
            # data is a precomputed distance matrix
            # to check the data is a valid distance matrix?
            return data

        # todo: the pdist may be too slow, to speed up for big data...
        dist_vec = distance.pdist(data, metric=self.name)
        dist_matrix = distance.squareform(dist_vec)
        return dist_matrix
