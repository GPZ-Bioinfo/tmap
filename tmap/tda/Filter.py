# -*- coding: utf-8 -*-
import numpy as np
import umap
from scipy.spatial.distance import pdist, squareform
from sklearn import decomposition, manifold

from tmap.tda.metric import Metric


_METRIC_BUILT_IN = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
                    "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto",
                    "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]


class Filters(object):
    """
    filters for TDA Mapper, to project data points onto a low dimensional space
    """

    def __init__(self, components=[0, 1], metric=None):
        if len(components) == 0:
            raise Exception("At least one filter is needed: %s" % str(components))
        self.components = components
        self.metric = metric

    def _check_data(self, data):
        # trivial check of data sanity
        if data is None:
            raise Exception("Data must not be None.")
        if type(data) is not np.ndarray:
            data = np.array(data)
        return data

    def fit_transform(self, data):
        # the base implementation of filters
        # just return the selected components
        data = self._check_data(data)
        assert self.metric is None
        return data[:, self.components]


class L1Centrality(Filters):
    """
    L1 Centrality filters.
    """

    def __init__(self, metric=Metric(metric="euclidean")):
        # default metric: euclidean
        # components is of 1-D
        super(L1Centrality, self).__init__(components=[0], metric=metric)

    def fit_transform(self, data):
        """
        Project data onto a L1 centrality axis.
        Params:
            data: data points or distance matrix.
        """
        data = self._check_data(data)
        dist_matrix = self.metric.fit_transform(data)
        return np.sum(dist_matrix, axis=1).reshape(data.shape[0], 1)


class LinfCentrality(Filters):
    """
    L-infinity centrality filters.
    """

    def __init__(self, metric=Metric(metric="euclidean")):
        # default metric: euclidean
        # components is of 1-D
        super(LinfCentrality, self).__init__(components=[0], metric=metric)

    def fit_transform(self, data):
        """
        Project data onto a L-inf Centrality axis.
        Params:
            data: data points or distance matrix.
        """
        data = self._check_data(data)
        dist_matrix = self.metric.fit_transform(data)
        return np.max(dist_matrix, axis=1).reshape(data.shape[0], 1)


class GaussianDensity(Filters):
    """
    Gaussian density filters.
    Params:
        h: The width of kernel.
    """

    def __init__(self, metric=Metric(metric="euclidean"), h=0.3):
        # default metric: euclidean
        # components is of 1-D
        super(GaussianDensity, self).__init__(components=[0], metric=metric)
        if h == 0:
            raise Exception("Parameter h must not be zero.")
        self.h = h

    def fit_transform(self, data):
        """
        Project data onto a Gaussian Density axis.
        Params:
            data: data points or distance matrix.
        """
        data = self._check_data(data)
        dist_matrix = self.metric.fit_transform(data)
        return np.sum(np.exp(-(dist_matrix ** 2 / (2 * self.h))), axis=1).reshape(data.shape[0], 1)


class PCA(Filters):
    """
    PCA filters.
    Params:
        components: The axis of projection. If you use components 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1], random_state=None, **kwds):
        # PCA only accept raw data and calculate euclidean distance "internally"
        super(PCA, self).__init__(components=components, metric=None)
        self.pca = decomposition.PCA(n_components=max(self.components) + 1, random_state=random_state, **kwds)

    def fit_transform(self, data):
        """
        Project data onto PCA components.
        Params:
            data: raw data points.
        """
        data = self._check_data(data)
        projected_data = self.pca.fit_transform(data)
        self.contribution_ratio = self.pca.explained_variance_ratio_[self.components]
        self.axis = self.pca.components_[self.components, :]
        return projected_data[:, self.components]


class TSNE(Filters):
    """
    TSNE filters.
    Params:
        components: The axis of projection. If you use components 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1], metric=Metric(metric="euclidean"), **kwds):
        super(TSNE, self).__init__(components=components, metric=metric)

        if self.metric.name in _METRIC_BUILT_IN:
            self.tsne = manifold.TSNE(n_components=max(self.components) + 1, metric=self.metric.name, **kwds)
        else:
            self.tsne = manifold.TSNE(n_components=max(self.components) + 1, metric="precomputed", **kwds)

    def fit_transform(self, data):
        """
        Project data onto TSNE axis.
        Params:
            data: data points or distance matrix.
        """
        data = self._check_data(data)
        if self.metric.name not in _METRIC_BUILT_IN:
            data = self.metric.fit_transform(data)
        projected_data = self.tsne.fit_transform(data)
        return projected_data[:, self.components]


class MDS(Filters):
    """
    MDS filters.
    Implements with sklearn.maniford.MDS
    Params:
        components: The axis of projection. If you use components 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1], metric=Metric(metric="euclidean"), **kwds):
        super(MDS, self).__init__(components=components, metric=metric)

        if self.metric.name == "euclidean":
            self.mds = manifold.MDS(n_components=max(self.components) + 1,
                                    dissimilarity="euclidean", n_jobs=-1, **kwds)
        else:
            self.mds = manifold.MDS(n_components=max(self.components) + 1,
                                    dissimilarity="precomputed", n_jobs=-1, **kwds)

    def fit_transform(self, data):
        data = self._check_data(data)
        if self.metric.name != "euclidean" and self.metric.name != "precomputed":
            data = squareform(pdist(data, metric=self.metric.name))
            data = self.metric.fit_transform(data)
        else:
            data = self.metric.fit_transform(data)
        projected_data = self.mds.fit_transform(data)
        return projected_data[:, self.components]


class PCOA(Filters):

    """
    PCoA filters.
    Implements with skbio.stats.ordination.pcoa
    Params:
        components: The axis of projection. If you use components 0 and 1, this is [0, 1].
    """

    def __init__(self, metric=Metric(metric="euclidean"), **kwds):
        super(PCOA, self).__init__()
        self.metric = metric

    def fit_transform(self, data):
        # try:
        from skbio.stats.ordination import pcoa
        # except:
        #     sys.exit("PCOA is using the pcoa implemened at scikit-bio.")
        data = self._check_data(data)
        if self.metric.name != "precomputed":
            data = squareform(pdist(data, metric=self.metric.name))
            data = self.metric.fit_transform(data)
        else:
            data = self.metric.fit_transform(data)

        projected_data = pcoa(data)
        self.pcoa = projected_data
        return self.pcoa.samples.values[:, self.components]


class UMAP(Filters):
    """
    MDS filters.
    Params:
        components: The axis of projection. If you use components 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1], metric=Metric(metric="euclidean"), **kwds):
        super(UMAP, self).__init__(components=components, metric=metric)

        if self.metric.name in _METRIC_BUILT_IN:
            self.umap = umap.UMAP(n_components=max(self.components) + 1,
                                  metric=self.metric.name, **kwds)
        else:
            self.umap = umap.UMAP(n_components=max(self.components) + 1,
                                  metric="precomputed", **kwds)

    def fit_transform(self, data):
        data = self._check_data(data)
        if self.metric.name not in _METRIC_BUILT_IN:
            data = self.metric.fit_transform(data)
        projected_data = self.umap.fit_transform(data)
        return projected_data[:, self.components]


_filter_dict = {'L1Centrality': L1Centrality,
                'LinfCentrality': LinfCentrality,
                'GaussianDensity': GaussianDensity,
                'PCA': PCA,
                'TSNE': TSNE,
                'MDS': MDS,
                'PCOA': PCOA,
                'UMAP': UMAP}
