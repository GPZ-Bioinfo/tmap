#! /usr/bin/python3
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from tmap.api.general import *
from tmap.tda import mapper
from tmap.tda.Filter import _filter_dict
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = 1000

import time
import argparse
import warnings

warnings.filterwarnings('ignore')


def cal_dis(data, metric="braycurtis", verbose=1):
    t1 = time.time()
    logger('Start calculating the [%s] distance matrix of input data.....' % metric, verbose=verbose)
    dis = squareform(pdist(data, metric=metric))
    logger('Complete the calculation of distance matrix. Take ', time.time() - t1, verbose=verbose)
    if 'index' in dir(data):
        dis = pd.DataFrame(dis, index=data.index, columns=data.index)
    else:
        dis = pd.DataFrame(dis)
    return dis


def generate_graph(input_data, dis=None, _eu_dm=None, eps_threshold=95, overlap=0.75, min_samples=3, r=40, filter='PCOA', verbose=1):
    if filter not in _filter_dict:
        logger("Wrong filter you provide, available fitler are", ','.join(_filter_dict.keys()), verbose=1)
        return
    else:
        filter = _filter_dict[filter]
    tm = mapper.Mapper(verbose=verbose)
    t1 = time.time()
    metric = Metric(metric="precomputed")
    lens = [filter(components=[0, 1], metric=metric, random_state=100)]
    projected_X = tm.filter(dis, lens=lens)
    logger("projection takes: ", time.time() - t1, verbose=verbose)
    ###
    t1 = time.time()
    eps = optimize_dbscan_eps(input_data, threshold=eps_threshold, dm=_eu_dm)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=r, overlap=overlap)
    graph = tm.map(data=input_data, cover=cover, clusterer=clusterer)
    logger(graph.info(), verbose=verbose)
    logger("graph generator take: ", time.time() - t1, verbose=verbose)
    return graph


def main(input, output, dis=None, _eu_dm=None, metric="braycurtis", eps=95, overlap=0.75, min_s=3, r=40, filter='PCOA', method='pickle', filetype='csv', verbose=1):
    data = data_parser(input, ft=filetype)
    if dis is None:
        dis = cal_dis(data, metric=metric, verbose=verbose)
    else:
        dis = pd.read_csv(dis, sep=',', index_col=0)

    if _eu_dm is None:
        eu_dm = cal_dis(data, metric='euclidean', verbose=1)
    else:
        eu_dm = pd.read_csv(_eu_dm, sep=',', index_col=0)

    if filter not in _filter_dict:
        logger("Wrong filter you provide, available fitler are ", ','.join(_filter_dict.keys()), verbose=1)
    graph = generate_graph(data,
                           dis=dis,
                           _eu_dm=eu_dm,
                           eps_threshold=eps,
                           overlap=overlap,
                           min_samples=min_s,
                           r=r,
                           filter=filter)

    if graph is None:
        logger("Empty graph generated... ERROR occur", verbose=1)
    else:
        # todo : implement a jsonable ndarray.
        graph.write(output)
        logger("Graph has been generated and stored into ", output,verbose=1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input", help="input data, normally formatted as row(sample) and columns(OTU/sOTU/other features)",
                        required=True)
    parser.add_argument("-O", "--output", help="Output File. Generated Network with its metadata.(python3's pickle format)",
                        required=True)
    parser.add_argument("-d", "--dis", help="Distance matrix of input file for lens. (Optional),If you doesn't provide, it will automatically \
                                             calculate distance matrix according to the file you provide and the metric you assign.",
                        default=None)
    parser.add_argument("--eu_dis", help="Eucildean Distance matrix of input file for clustering. (Optional),If you doesn't provide, it will automatically \
                                             calculate distance matrix according to the file you provide and the metric you assign.",
                        default=None)
    parser.add_argument("-m", "--metric", help="""The distance metric to use.The distance function can \
                                                  be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                                  'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                                                  'jaccard', 'kulsinski', 'mahalanobis', 'matching',
                                                  'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                                                  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.""",
                        default='braycurtis', type=str)
    parser.add_argument("-ft", "--file_type", help="File type of metadata you provide [csv|xlsx]. Separtor could be tab, comma, or others.",
                        type=str, default='csv')
    parser.add_argument("-eps", "--eps_threshold", dest='eps', help="The threshold which used to identify suitable eps for DBSCAN. \
                                                         It is a measurement of a percentage among the pairwise distances. It need to be interget,",
                        type=int, default=95)
    parser.add_argument("-ol", "--overlap", help="It decides the level of the expansion of each partition. must must greater than 0.",
                        type=float, default=0.75)
    parser.add_argument("-ms", "--min_samples", dest='ms', help="The number of samples (or total weight) in a neighborhood for a point to \
                                                     be considered as a core point. This includes the point itself.",
                        type=int, default=3)
    parser.add_argument("-r", "--resolution", help="It decides the number of partition of each axis at Cover.",
                        type=int, default=40)
    parser.add_argument("-f", "--filter", help="The type of filter to transform data. The available filter are \
                                            'L1Centrality','LinfCentrality','GaussianDensity','PCA','TSNE','MDS',\
                                            'PCOA','UMAP'. Default is [PCOA].",
                        type=str, default='PCOA')
    parser.add_argument("--method", help="Method for store output file. [pickle] other method will be implemented at future.",
                        type=str, default='pickle')
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    verbose = args.verbose
    dis = args.dis
    metric = args.metric

    if (dis is not None and metric != 'braycurtis'):
        logger("Distance matrix is given, assigned metric({m}) for calculating distance metric is useless.".format(m=metric), verbose=verbose)

    process_output(output=args.output)

    main(input=args.input,
         output=args.output,
         dis=dis,
         _eu_dm=args.eu_dis,
         metric=metric,
         eps=args.eps,
         overlap=args.overlap,
         min_s=args.ms,
         r=args.resolution,
         filter=args.filter,
         filetype=args.file_type,
         verbose=verbose)
