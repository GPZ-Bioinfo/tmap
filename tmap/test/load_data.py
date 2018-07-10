import pandas as pd
import os

base_path = os.path.abspath(__file__)

def FGFP_genus_profile():
    X = pd.read_csv(os.path.join(os.path.dirname(base_path), "test_data", "FGFP_genus_data.csv"),
                    index_col=0, header=0, sep=',')
    # normalize the taxa counts as relative abundances
    X = X.div(X.sum(axis=1), axis=0)
    return X

def FGFP_metadata():
    metadata = pd.read_csv(os.path.join(os.path.dirname(base_path), "test_data", 'FGFP_metadata.tsv'),
                           sep='\t', index_col=0, header=0)
    return metadata


def FGFP_BC_dist():
    dm = pd.read_csv(os.path.join(os.path.dirname(base_path), "test_data", "FGFP_bc_dm.csv"),
                     index_col=0, header=0, sep=',')
    return dm
