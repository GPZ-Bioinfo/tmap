import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
import numpy as np
import time,argparse,os
from tmap.api.general import data_parser,write_data,randomString,logger,preprocess_metadata_beta
from scipy.spatial.distance import squareform, pdist

importr("vegan")

random_str = randomString(10)
_static_data = './%s.envfit.data' % random_str
_static_dis = './%s.envfit.dis'% random_str
_static_metadata = './%s.envfit.metadata'% random_str

def prepare(input,metadata,dis,metric,filetype):
    data = data_parser(input, ft=filetype)
    if dis is None:
        dis = squareform(pdist(data,metric=metric))
        dis = pd.DataFrame(dis,index=data.index,columns=data.index)
    else:
        dis = data_parser(dis,ft=filetype)
    metadata = data_parser(metadata,ft=filetype)
    # preprocess metadata
    metadata = preprocess_metadata_beta(data,metadata,verbose=1)

    dis.to_csv(_static_dis, sep=',', index=1)
    data.to_csv(_static_data,sep=',',index=1)
    metadata.to_csv(_static_metadata,sep=',',index=1)

def envfit_metadata(data_path ,metadata_path ,dist_path,n_iter=500,return_ord = False):

    rcode = """
    genus_table <- read.csv('{path_data}',row.names = 1,check.names=FALSE)
    metadata <- read.csv('{path_metadata}',row.names = 1,check.names=FALSE)
    dist <- read.csv('{path_dist}',row.names = 1,check.names=FALSE)
    dist <- as.dist(dist)
    ord <- capscale(dist ~ -1)
    """.format(path_data=data_path ,
               path_dist = dist_path,
               path_metadata=metadata_path)
    robjects.r(rcode)

    envfit_result = robjects.r(
        """
        fit <- envfit(ord,metadata,permutations = {n_iter})
        fit$vectors
        """.format(n_iter=n_iter))
    R_ord = robjects.r('summary(ord)$sites')
    R_pro_X_df = pd.DataFrame(data=np.array(R_ord) ,index=R_ord.rownames ,columns=R_ord.colnames)
    fit_result = pd.DataFrame(columns=["r2" ,"pvals" ,"Source" ,"End"]
                              ,index=envfit_result[envfit_result.names.index("arrows")].rownames)
    fit_result.loc[: ,"r2"] = envfit_result[envfit_result.names.index("r")]
    fit_result.loc[:, "pvals"] = envfit_result[envfit_result.names.index("pvals")]
    fit_result.loc[:, ["Source" ,"End"]] = np.array(envfit_result[envfit_result.names.index("arrows")])
    if return_ord:
        return fit_result ,R_pro_X_df
    else:
        return fit_result

def main(input,metadata,dis,output,metric,n_iter,filetype,keep=False,verbose=1):
    logger("prepare the input data for envfit......",verbose=verbose)
    prepare(input,metadata,dis,metric,filetype)
    logger("Start to load data into r environment and start envfit...",verbose=verbose)
    t1 = time.time()
    fit_result = envfit_metadata(data_path=_static_data,
                    metadata_path=_static_metadata,
                    dist_path=_static_dis,
                    n_iter=n_iter,
                    return_ord=False)
    logger("Finish envfit, take", time.time()-t1,'second',verbose=verbose)
    write_data(fit_result,output)

    if not keep:
        os.remove(_static_dis)
        os.remove(_static_data)
        os.remove(_static_metadata)

if __name__ == '__main__':

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
                        default='braycurtis',type=str)
    parser.add_argument("-ft", "--file_type", help="File type of metadata you provide [csv|xlsx]. Separtor could be tab, comma, or others.",
                        type=str,default='csv')
    parser.add_argument( "--keep", help="Keep intermediate files.",
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

    main(input=input,
         metadata=metadata,
         dis=dis,
         output=output,
         metric=metric,
         n_iter=n_iter,
         filetype=args.file_type,
         keep=args.keep,
         verbose=args.verbose)
