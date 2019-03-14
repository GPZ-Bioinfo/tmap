#! /usr/bin/python3
import argparse
import os
import time

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from scipy.spatial.distance import squareform, pdist

from tmap.api.general import *


def prepare(input, output,metadata, dis, metric, filetype):
    data = data_parser(input, ft=filetype)
    if dis is None:
        dis = squareform(pdist(data, metric=metric))
        dis = pd.DataFrame(dis, index=data.index, columns=data.index)
    else:
        dis = data_parser(dis, ft=filetype)
    metadata = data_parser(metadata, ft=filetype)
    # preprocess metadata
    post_metadata = process_metadata_beta(data, metadata, verbose=1)


    dir_path = os.path.dirname(os.path.realpath(output))
    logger("Output temp file into %s" % _static_dis.format(output=dir_path).strip('.envfit.data'), verbose=1)
    dis.to_csv(_static_dis.format(output=dir_path), sep=',', index=1)
    data.to_csv(_static_data.format(output=dir_path), sep=',', index=1)
    if post_metadata.shape[1] / metadata.shape[1] >= 5:
        logger("May occur error because of process metadata, it may check carefully... It may wrongly take numerical columns as categorical columns so make dimension explosion. ", verbose=1)
        metadata.to_csv(_static_beforemetadata.format(output=dir_path), sep=',', index=1)
        post_metadata.to_csv(_static_metadata.format(output=dir_path), sep=',', index=1)
    else:
        metadata.to_csv(_static_beforemetadata.format(output=dir_path), sep=',', index=1)
        post_metadata.to_csv(_static_metadata.format(output=dir_path), sep=',', index=1)




def envfit_metadata(data_path, metadata_path, dist_path, n_iter=500, return_ord=False):
    rcode = """
    genus_table <- read.csv('{path_data}',row.names = 1,check.names=FALSE)
    metadata <- read.csv('{path_metadata}',row.names = 1,check.names=FALSE)
    dist <- read.csv('{path_dist}',row.names = 1,check.names=FALSE)
    dist <- as.dist(dist)
    ord <- capscale(dist ~ -1)
    """.format(path_data=data_path,
               path_dist=dist_path,
               path_metadata=metadata_path)
    robjects.r(rcode)

    envfit_result = robjects.r(
        """
        fit <- envfit(ord,metadata,permutations = {n_iter})
        fit$vectors
        """.format(n_iter=n_iter))
    R_ord = robjects.r('summary(ord)$sites')
    R_pro_X_df = pd.DataFrame(data=np.array(R_ord), index=R_ord.rownames, columns=R_ord.colnames)
    fit_result = pd.DataFrame(columns=["r2", "pvals", "Source", "End"]
                              , index=envfit_result[envfit_result.names.index("arrows")].rownames)
    fit_result.loc[:, "r2"] = envfit_result[envfit_result.names.index("r")]
    fit_result.loc[:, "pvals"] = envfit_result[envfit_result.names.index("pvals")]
    fit_result.loc[:, ["Source", "End"]] = np.array(envfit_result[envfit_result.names.index("arrows")])
    if return_ord:
        return fit_result, R_pro_X_df
    else:
        return fit_result


def main(input, metadata, dis, output, metric, n_iter, filetype, just_pre=False, keep=True, verbose=1):
    """
    :param input:  file path of input data. It should be csv format or tab format instead of XLSX. Row represents samples, Column represents features(Mostly it may be OTU/Genus).
    :param metadata: file path of metadata data. The number of row must equal to the number of row in `input`. It also should be csv format.
    :param output: path of output file. (FULL name instead of prefix. Only one output.)
    :param dis: Pairwise distance matrix. (It could be none and use `metric` to let the programme calculated for user.)
    :param n_iter: The number of time to shuffle at `envfit`.
    :param metric: The distance metric to use.
    :param just_pre: Boolean value for indicate whether to run envfit or just use this function to preprocess metadata.
    :param keep:
    :param verbose:
    :return:
    """
    logger("prepare the input data for envfit......", verbose=verbose)
    prepare(input, output,metadata, dis, metric, filetype)
    logger("Start to load data into r environment and start envfit...", verbose=verbose)
    t1 = time.time()
    dir_path = os.path.dirname(os.path.realpath(output))
    if not just_pre:
        fit_result = envfit_metadata(data_path=_static_data.format(output=dir_path),
                                     metadata_path=_static_metadata.format(output=dir_path),
                                     dist_path=_static_dis.format(output=dir_path),
                                     n_iter=n_iter,
                                     return_ord=False)
        logger("Finish envfit, take", time.time() - t1, 'second', verbose=verbose)
        write_data(fit_result, output)

    if not keep:
        logger("removing files...")
        os.remove(_static_dis.format(output=dir_path))
        os.remove(_static_data.format(output=dir_path))
        os.remove(_static_metadata.format(output=dir_path))
        os.remove(_static_beforemetadata.format(output=dir_path))


if __name__ == '__main__':
    importr("vegan")

    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input", help="input data, normally formatted as row(sample) and columns(OTU/sOTU/other features)",
                        required=True)
    parser.add_argument("-M", "--metadata", help="Metadata files need to calculate the envfit.",
                        required=True)
    parser.add_argument("-O", "--output", help="Prefix of output File. Envfit output result",
                        required=True)
    parser.add_argument("-d", "--dis", help="Distance matrix of input file. (Optional),If you doesn't provide, it will automatically \
                                             calculate distance matrix according to the file you provide and the metric you assign.",
                        default=None)
    parser.add_argument("-i", "--iter", help="The number of times to shuffle for calculating envfit. [1000]",
                        default=1000, type=int)

    parser.add_argument("-m", "--metric", help="""The distance metric to use.The distance function can \
                                                  be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                                  'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                                                  'jaccard', 'kulsinski', 'mahalanobis', 'matching',
                                                  'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                                                  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.""",
                        default='braycurtis', type=str)
    parser.add_argument("-ft", "--file_type", help="File type of metadata you provide [csv|xlsx]. Separtor could be tab, comma, or others.",
                        type=str, default='csv')
    parser.add_argument("-tn", "--temp_name", help="Manually assign name to temporal files.",
                        type=str, default='')
    parser.add_argument("--dont_analysis", dest='just_pre',help="Don not run envfit, just preprocess the metadata.",
                        action="store_true")
    parser.add_argument("--keep", help="Keep intermediate files.",
                        action="store_true")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    input = args.input
    metadata = args.metadata
    output = args.output
    dis = args.dis
    metric = args.metric
    n_iter = args.iter

    random_str = randomString(10)
    if args.temp_name:
        _static_data = '{output}/%s.envfit.data' % args.temp_name
        _static_dis = '{output}/%s.envfit.dis' % args.temp_name
        _static_metadata = '{output}/%s.envfit.metadata' % args.temp_name
        _static_beforemetadata = '{output}/%s.envfit.raw_metadata' % args.temp_name
    else:
        _static_data = '{output}/%s.envfit.data' % random_str
        _static_dis = '{output}/%s.envfit.dis' % random_str
        _static_metadata = '{output}/%s.envfit.metadata' % random_str
        _static_beforemetadata = '{output}/%s.envfit.raw_metadata' % random_str

    process_output(output=output)
    main(input=input,
         metadata=metadata,
         dis=dis,
         output=output,
         metric=metric,
         n_iter=n_iter,
         filetype=args.file_type,
         just_pre=args.just_pre,
         keep=args.keep,
         verbose=args.verbose)
