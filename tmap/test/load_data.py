import pandas as pd
import numpy as np
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

def FGFP_metadata_ready():
    metadata = pd.read_csv(os.path.join(os.path.dirname(base_path), "test_data", 'FGFP_metadata_ready.tsv'),
                           sep='\t', index_col=0, header=0)
    return metadata

def Daily_genus_profile(type):
    if type.lower() == "stool":
        X = pd.read_csv(os.path.join(os.path.dirname(base_path), "test_data", 'Daily_stool_genus.csv'),
                               sep=',', index_col=0, header=0)
        X = X.div(X.sum(1), axis=0)
        return X
    if type.lower() == "saliva":
        X = pd.read_csv(os.path.join(os.path.dirname(base_path), "test_data", 'Daily_saliva_genus.csv'),
                               sep=',', index_col=0, header=0)
        X = X.div(X.sum(1), axis=0)
        return X

def Daily_metadata_ready():
    metadata = pd.read_csv(os.path.join(os.path.dirname(base_path), "test_data", 'Daily_metadata.csv'),
                           sep=',', index_col=0, header=0)
    return metadata

#
# def FGFP_metadata_process():
#     metadata = pd.read_csv(os.path.join(os.path.dirname(base_path), "test_data", 'FGFP_metadata.tsv'),
#                            sep='\t', index_col=0, header=0)
#     meta_metadata = pd.read_excel(os.path.join(os.path.dirname(base_path), "test_data","Supplementary_Table1.xlsx"))
#     meta_metadata = meta_metadata.set_index('Variable')
#
#     # mapping the variables
#     order_new_columns = []
#     for i in metadata.columns:
#         if i not in meta_metadata.index:
#             cache = list(i)
#             cache = sorted([(len(set(cache).intersection(set(list(_)))), _, i) for _ in meta_metadata.index], reverse=True)
#             if cache[0][0] == cache[1][0]:
#                 max_one = [_ for _ in cache if _[0] == cache[0][0]]
#                 max_one = [(_[0] / len(set(_[1])), _[1], _[2]) for _ in max_one]
#                 order_new_columns.append(sorted(max_one)[-1][1])
#             else:
#                 order_new_columns.append(cache[0][1])
#         else:
#             order_new_columns.append(i)
#
#     meta_metadata_type = meta_metadata.loc[order_new_columns, 'Type']
#
#     numerical_feas = [_ for _, v in zip(metadata.columns, meta_metadata_type) if v.strip(' ') == 'Numerical']
#     categorical_feas = [_ for _, v in zip(metadata.columns, meta_metadata_type) if v.strip(' ') == 'Categorical']
#     boolean_feas = [_ for _, v in zip(metadata.columns, meta_metadata_type) if v.strip(' ') == 'Categorical (logical)']
#
#     # remove boolean features with nan and place it in categorical features.
#     nan_boolean_fea = np.array(boolean_feas)[metadata.loc[:, boolean_feas].isna().any()]
#     for nan_boolean in list(nan_boolean_fea):
#         boolean_feas.remove(nan_boolean)
#         categorical_feas.append(nan_boolean)
#     ############################################################
#     # One-hot encoded
#     transformed_df = pd.DataFrame(index=metadata.index, )
#     subset_metadata = metadata.loc[:, categorical_feas].fillna('nan')
#     for each in categorical_feas:
#         all_classes = set(subset_metadata.loc[:, each])
#         for each_class in all_classes:
#             transformed_df.loc[:, '%s:%s' % (each, str(each_class))] = [1 if _ == each_class else 0 for _ in subset_metadata.loc[:, each]]
#     ############################################################
#     boolean_feas_df = metadata.loc[:, boolean_feas]
#     ############################################################
#     from sklearn.preprocessing import Imputer
#     imputer_ = Imputer(strategy='median')
#     numerical_df = pd.DataFrame(data=imputer_.fit_transform(metadata.loc[:, numerical_feas]), index=metadata.index, columns=numerical_feas)
#     ############################################################
#
#     metadata_transformed = pd.concat([numerical_df, boolean_feas_df, transformed_df], axis=1)
#     return metadata_transformed