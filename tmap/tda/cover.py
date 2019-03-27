from __future__ import print_function
import numpy as np
import itertools


class Cover(object):
    """
    Covering the projection data

    :param numpy.ndarray/pandas.DataFrame projected_data:

            Normally, `projected_data` should be the data transformed by *MDS* or *t-SNE* or *PCA*. It decides the way of partition for the original point cloud.

    :param integer resolution:

            It decides the number of partition of each axis at ``projected_data``

    :param float overlap:

            `overlap` must greater than 0. It decides the level of the expansion of each partition. If `overlap` equals to 0.5, each partition (include the first and the last) will expand 0.5 times of orginal.

    """
    def __init__(self, projected_data, resolution=10, overlap=0.5):
        """
        :param projected_data: the projection data used for covering
        :param resolution: resolution of covering
        :param overlap: overlap of adjacent covers
        """
        self.resolution = resolution
        self.overlap = overlap
        if overlap <= 0:
            print("overlap must greater than 0")
            exit(1)
        self.n_points, self.n_dimensions = projected_data.shape
        self.data = projected_data
        # upper and lower bounds, chunk and overlap for each dimension of the projected space
        self.floor, self.roof = (np.min(projected_data, axis=0), np.max(projected_data, axis=0))
        self.chunk_width = (self.roof - self.floor) / resolution
        self.overlap_width = self.chunk_width * overlap

    @property
    def hypercubes(self):
        """
        Generate hypercubes (covering) using a generator function

        :return: It returns a mask for the projected_data. Each row is a list of boolean for indicating which samples are within the current cube. The row number of hypercubes represents the number of partition.
        :rtype: numpy.ndarray
        """
        return self._get_hypercubes()

    def _get_hypercubes(self,output_bounds=False):
        # generate hypercube index based on the resolution parameter (how many and where the hypercube is?)
        bins = itertools.product(np.arange(self.resolution), repeat=self.n_dimensions)
        n_bins = self.resolution**self.n_dimensions
        hypercubes = np.zeros((n_bins, self.n_points), dtype=bool)
        bounds_with_overlap = []
        bounds_without_overlap = []
        for i, bin in enumerate(bins):
            lower_bound = self.floor + bin * self.chunk_width
            upper_bound = lower_bound + self.chunk_width + self.overlap_width
            lower_bound -= self.overlap_width
            mask = np.all((self.data >= lower_bound) & (self.data < upper_bound), axis=1)
            hypercubes[i, :] = mask
            bounds_with_overlap.append((lower_bound,upper_bound))
            bounds_without_overlap.append((lower_bound + self.overlap_width,upper_bound - self.overlap_width))
        if output_bounds:
            return bounds_with_overlap,bounds_without_overlap
        else:
            return hypercubes
