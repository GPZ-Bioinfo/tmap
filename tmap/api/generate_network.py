import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from tmap.api.general import logger
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps, cover_ratio,

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = 1000
import warnings
import time
import argparse

warnings.filterwarnings('ignore')

global_verbose = 0
def parser(path, ft='csv',verbose=1,**kwargs):
    if ft == 'csv':
        df = pd.read_csv(path, index_col=0, header=True, **kwargs)
    else:
        df = pd.read_excel(path, index_col=0, header=True, **kwargs)
    logger('Input data path: ', path,verbose=verbose)
    logger('Shape of Input data: ', df.shape,verbose=verbose)
    return df


def cal_dis(data, metric="braycurtis", verbose=1):
    t1 = time.time()
    logger('Start calculating the distance matrix of input data.....',verbose=verbose)
    dis = squareform(pdist(data, metric=metric))
    logger('Complete the calculation of distance matrix. Take ',t1-time.time(),verbose=verbose)
    return dis


def generate_graph(input_data, dis, _eu_dm=None, eps_threshold=95, overlap=0.75, min_samples=3, r="auto", filter_=Filter.PCOA, verbose=1):
    tm = mapper.Mapper(verbose=verbose)
    # TDA Step2. Projection
    t1 = time.time()
    metric = Metric(metric="precomputed")
    lens = [filter_(components=[0, 1], metric=metric, random_state=100)]
    projected_X = tm.filter(dis, lens=lens)
    logger("projection takes: ", time.time() - t1, verbose=verbose)
    ###
    t1 = time.time()
    eps = optimize_dbscan_eps(input_data, threshold=eps_threshold, dm=_eu_dm)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_X), resolution=r, overlap=overlap)
    graph = tm.map(data=input_data, cover=cover, clusterer=clusterer)
    logger('Graph covers %.2f percentage of samples.' % cover_ratio(graph, input_data),verbose=verbose)
    logger("graph time: ", time.time() - t1,verbose=verbose)

    graph_name = "{eps}_{overlap}_{r}_{filter}.graph".format(eps=eps_threshold, overlap=overlap, r=r, filter=lens[0].__class__.__name__)
    return graph

def main(input,output,dis,metric,eps,overlap,min,r,filter,verbose):
    data = parser(input,)
    if dis is not None:

    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input", help="input data, normally formatted as row(sample) and columns(OTU/sOTU/other features)",
                        required=True)
    parser.add_argument("-O", "--output", help="Output File. Generated Network with its metadata.(Json format)",
                        required=True)
    parser.add_argument("-d", "--dis", help="Distance matrix of input file. (Optional),If you doesn't provide, it will automatically \
                                             calculate distance matrix according to the file you provide and the metric you assign.",
                        default=None)
    parser.add_argument("-m", "--metric", help="""The distance metric to use.The distance function can \
                                                  be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                                  'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                                                  'jaccard', 'kulsinski', 'mahalanobis', 'matching',
                                                  'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                                                  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.""",
                        default='braycurtis',type=str)
    parser.add_argument("-sep", "--sep", help="Separator ",
                        type=str,default=',')
    parser.add_argument("-eps", "--eps_threshold", help="The threshold which used to identify suitable eps for DBSCAN. \
                                                         It is a measurement of a percentage among the pairwise distances. It need to be interget,",
                        type=int)
    parser.add_argument("-ol", "--overlap", help="It decides the level of the expansion of each partition. must must greater than 0.",
                        type=float)
    parser.add_argument("-ms", "--min_samples", help="The number of samples (or total weight) in a neighborhood for a point to \
                                                     be considered as a core point. This includes the point itself.",
                        type=int)
    parser.add_argument("-r", "--resolution", help="It decides the number of partition of each axis at Cover.",
                        type=int)
    parser.add_argument("-f", "--filter", help="The distance metric to use.The distance function can ",
                        type=str)
    parser.add_argument("-V", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    verbose = args.verbose
    dis = args.dis
    metric = args.metric

    if (dis is not None and metric != 'braycurtis'):
        logger("Distance matrix is given, assigned metric({m}) for calculating distance metric is useless.".format(m=metric),verbose=verbose)

    main(input=args.input,
         output=args.output,
         dis=dis,
         metric=metric,
         eps=args.eps_threhold,
         overlap=args.overlap,
         min=args.min_samples,
         r=args.resolution,
         filter=args.filter,
         verbose=verbose)
