from tmap.netx.SAFE import get_enriched_nodes
import networkx as nx
from tqdm import tqdm
import scipy.stats as scs
import numpy as np
import pandas as pd


def get_component_nodes(fea, enriched_nodes, graph):
    sub_nodes = list(set(enriched_nodes.get(fea, [])))
    sub_edges = [edge for edge in graph['edges'] if edge[0] in sub_nodes and edge[1] in sub_nodes]

    G = nx.Graph()
    G.add_nodes_from(sub_nodes)
    G.add_edges_from(sub_edges)

    comp_nodes = [nodes for nodes in nx.algorithms.components.connected_components(G)]
    return comp_nodes

def is_enriched(s1,s2,s3,s4):
    total_1 = len(s1) + len(s2)
    total_2 = len(s3) + len(s4)
    if total_1==0 or total_2 ==0:
        return False
    if len(s1)/total_1 > len(s3)/total_2:
        return True
    else:
        return False

def build_network(fea, graph, safe_scores, n_iter=5000, p_value=0.05,enriched_centroides=None,_mode='single'):
    if _mode not in ['all','single']:
        print('SyntaxError...')
        return
    global_correlative_feas = {}
    sub_correlative_feas = {}
    metainfo = {}
    total_nodes = set(list(safe_scores.values())[0].keys())
    print('building network...')
    if fea in safe_scores.keys():
        if not enriched_centroides:
            min_p_value = 1.0 / (n_iter + 1.0)
            threshold = np.log10(p_value) / np.log10(min_p_value)
            enriched_centroides, enriched_nodes = get_enriched_nodes(pd.DataFrame.from_dict(safe_scores, orient='index'), threshold, graph, centroids=True)

        comp_nodes = get_component_nodes(fea, enriched_centroides, graph)
        # lefted_components = [nodes for nodes in comp_nodes if len(nodes) >= components_samples_threshold]
        fea_enriched_nodes = set(enriched_centroides[fea])
        fea_nonenriched_nodes = total_nodes.difference(fea_enriched_nodes)
        metainfo[fea] = fea_enriched_nodes, comp_nodes
        for o_f in enriched_centroides.keys():
            if o_f != fea:
                #                           fea enriched nodes, fea non-enriched_nodes
                # o_f_enriched_nodes           s1                        s2
                # o_f_non-enriched_nodes       s3                        s4
                o_f_enriched_nodes = set(enriched_centroides[o_f])
                o_f_nonenriched_nodes = total_nodes.difference(o_f_enriched_nodes)

                s1 = o_f_enriched_nodes.intersection(fea_enriched_nodes)
                s2 = o_f_enriched_nodes.intersection(fea_nonenriched_nodes)
                s3 = o_f_nonenriched_nodes.intersection(fea_enriched_nodes)
                s4 = o_f_nonenriched_nodes.intersection(fea_nonenriched_nodes)
                oddsratio, pvalue = scs.fisher_exact([[len(s1), len(s2)],
                                                      [len(s3), len(s4)]])


                if pvalue <= 0.05 and is_enriched(s1,s2,s3,s4):
                    global_correlative_feas[o_f] = (oddsratio, pvalue)
                    metainfo[o_f] = (s1, s2, s3, s4)

                for idx, nodes in enumerate(comp_nodes):
                    #                           fea this comp enriched nodes, fea other comp enriched nodes
                    # o_f_enriched_nodes           s1                          s2
                    # o_f_non-enriched_nodes       s3                          s4
                    nodes = set(nodes)
                    _s1 = o_f_enriched_nodes.intersection(nodes)
                    _s2 = o_f_enriched_nodes.intersection(fea_enriched_nodes.difference(nodes))
                    _s3 = o_f_nonenriched_nodes.intersection(nodes)
                    _s4 = o_f_nonenriched_nodes.intersection(fea_enriched_nodes.difference(nodes))
                    oddsratio, pvalue1 = scs.fisher_exact([[len(_s1), len(_s2)],
                                                           [len(_s3), len(_s4)]])
                    #                           fea this comp enriched nodes, fea non-enriched nodes
                    # o_f_enriched_nodes           s1                          s2
                    # o_f_non-enriched_nodes       s3                          s4
                    s1 = o_f_enriched_nodes.intersection(nodes)
                    s2 = o_f_enriched_nodes.intersection(fea_nonenriched_nodes)
                    s3 = o_f_nonenriched_nodes.intersection(nodes)
                    s4 = o_f_nonenriched_nodes.intersection(fea_nonenriched_nodes)
                    oddsratio, pvalue2 = scs.fisher_exact([[len(s1), len(s2)],
                                                           [len(s3), len(s4)]])
                    if pvalue1 <= 0.05 and pvalue2 <= 0.05 and is_enriched(s1,s2,s3,s4) and is_enriched(_s1,_s2,_s3,_s4):
                        sub_correlative_feas[(idx, len(nodes), o_f)] = (pvalue1, pvalue2)
                        metainfo[(idx, len(nodes), o_f)] = (_s1, _s2, s2)

    return global_correlative_feas, sub_correlative_feas, metainfo

def construct_correlative_metadata(fea, global_correlative_feas, sub_correlative_feas, metainfo, node_data, nodes_metadata):
    print('processing correlative data......')
    # processing global correlative feas
    global_headers = ['other feature',
                      'fisher-exact test pvalue',
                      'ranksum in co-enriched nodes',
                      'ranksum in others nodes',
                      'coverage/%', ]

    sub_headers = ['n_comps',
                   'comps_size',
                   'other feature',
                   'Fisher test pvalue(co-enriched,enriched)',
                   'Fisher test pvalue(co-enriched,others)',
                   'coenriched-enriched pvalue',
                   'coenriched-others pvalue',
                   'enriched-others pvalue',
                   'coverage/%',
                   ]
    global_corr_df = pd.DataFrame(columns=global_headers)
    for o_f in list(global_correlative_feas.keys()):
        _1, f_p = global_correlative_feas[o_f]
        s1, s2, s3, s4 = metainfo[o_f]

        if o_f in node_data.columns:
            _data = node_data
        elif o_f in nodes_metadata.columns:
            _data = nodes_metadata
        else:
            print('error feature %s' % o_f)
            return
        y1 = _data.loc[s1, o_f]
        y2 = _data.loc[set.union(s2, s3, s4), o_f]

        if fea in node_data.columns:
            _data = node_data
        elif fea in nodes_metadata.columns:
            _data = nodes_metadata
        _y1 = _data.loc[s1, fea]
        _y2 = _data.loc[set.union(s2, s3, s4), fea]
        ranksum_p1 = scs.ranksums(y1, y2)[1]
        ranksum_p2 = scs.ranksums(_y1, _y2)[1]
        if not len(s1) + len(s3):
            coverage = np.nan
        else:
            coverage = ((len(s1) + len(s2)) / (len(s1) + len(s3))) * 100

        global_corr_df = global_corr_df.append(pd.DataFrame([[o_f,f_p ,ranksum_p1, ranksum_p2, coverage]], columns=global_headers))

    # processing subgraph correlative feas
    sub_corr_df = pd.DataFrame(columns=sub_headers)
    for n_c, size, o_f in sub_correlative_feas.keys():
        if o_f in node_data.columns:
            _data = node_data
        elif o_f in nodes_metadata.columns:
            _data = nodes_metadata
        else:
            print('error feature %s' % o_f)
            return
        coenriched_nodes, enriched_nodes_o_f_enriched, nonenriched_nodes_o_f_enriched = metainfo[(n_c, size, o_f)]
        f_p1,f_p2 = sub_correlative_feas[(n_c, size, o_f)]
        y1 = _data.loc[coenriched_nodes, o_f]
        y2 = _data.loc[enriched_nodes_o_f_enriched, o_f]
        y3 = _data.loc[nonenriched_nodes_o_f_enriched, o_f]
        p1 = scs.ranksums(y1, y2)[1]
        p2 = scs.ranksums(y1, y3)[1]
        p3 = scs.ranksums(y2, y3)[1]
        coverage = len(coenriched_nodes) / len(set.union(coenriched_nodes, enriched_nodes_o_f_enriched, nonenriched_nodes_o_f_enriched)) * 100
        sub_corr_df = sub_corr_df.append(pd.DataFrame([['comps%s' % n_c, size, o_f,f_p1,f_p2, p1, p2, p3, coverage]], columns=sub_headers))

    return global_corr_df, sub_corr_df

def cal_fisher_exact_dis(graph, safe_scores, n_iter=5000, p_value=0.05,enriched_centroides=None):
    dist_matrix = pd.DataFrame(data=np.nan,index=safe_scores.keys(),columns = safe_scores.keys())
    metainfo = {}
    total_nodes = set(list(safe_scores.values())[0].keys())
    print('building network...')
    if not enriched_centroides:
        min_p_value = 1.0 / (n_iter + 1.0)
        threshold = np.log10(p_value) / np.log10(min_p_value)
        enriched_centroides, enriched_nodes = get_enriched_nodes(pd.DataFrame.from_dict(safe_scores, orient='index'), threshold, graph, centroids=True)

    for fea in tqdm(safe_scores.keys()):
        comp_nodes = get_component_nodes(fea, enriched_centroides, graph)
        # lefted_components = [nodes for nodes in comp_nodes if len(nodes) >= components_samples_threshold]

        fea_enriched_nodes = set(enriched_centroides[fea])
        fea_nonenriched_nodes = total_nodes.difference(fea_enriched_nodes)
        metainfo[fea] = fea_enriched_nodes, comp_nodes
        for o_f in enriched_centroides.keys():
            if np.isnan(dist_matrix.loc[fea, o_f]):
                #                           fea enriched nodes, fea non-enriched_nodes
                # o_f_enriched_nodes           s1                        s2
                # o_f_non-enriched_nodes       s3                        s4
                o_f_enriched_nodes = set(enriched_centroides[o_f])
                o_f_nonenriched_nodes = total_nodes.difference(o_f_enriched_nodes)
                s1 = o_f_enriched_nodes.intersection(fea_enriched_nodes)
                s2 = o_f_enriched_nodes.intersection(fea_nonenriched_nodes)
                s3 = o_f_nonenriched_nodes.intersection(fea_enriched_nodes)
                s4 = o_f_nonenriched_nodes.intersection(fea_nonenriched_nodes)
                oddsratio, pvalue = scs.fisher_exact([[len(s1), len(s2)],
                                                      [len(s3), len(s4)]], alternative='greater')
                if is_enriched(s1,s2,s3,s4):
                    dist_matrix.loc[fea,o_f] = dist_matrix.loc[o_f,fea] = pvalue
                else:
                    dist_matrix.loc[fea, o_f] = dist_matrix.loc[o_f, fea] = 1
        dist_matrix.loc[fea,fea] = 0
    return dist_matrix


# if __name__ == '__main__':
#     from tmap.netx.SAFE import construct_node_data
#     import itertools
#
#     fea = 'Mean_corpuscular_hemoglobin_concentration'
#     node_data = construct_node_data(graph, input_otu)
#     nodes_metadata = construct_node_data(graph, metadata)
#     global_correlative_feas, sub_correlative_feas, metainfo = build_network(fea, graph, enriched_SAFE_total, p_value=0.05,enriched_centroides=)
#     global_corr_df, sub_corr_df = construct_correlative_metadata(fea, global_correlative_feas, sub_correlative_feas, metainfo, node_data, nodes_metadata)
#
#
#     global_corr_df.loc[global_corr_df.loc[:, 'ranksum in co-enriched nodes'] < 0.05, :].sort_values('fisher-exact test pvalue', ascending=True).to_csv(
#         os.path.join(base_path, '%s_global_corr.csv' % fea), index=False)
#     sub_corr_df.loc[(sub_corr_df.comps_size >= 10) & (sub_corr_df.loc[:, 'coenriched-others pvalue'] < 0.05), :].sort_values(['n_comps', 'coenriched-others pvalue'],
#                                                                                                                              ascending=True).to_csv(
#         os.path.join(base_path, '%s_sub_corr.csv' % fea), index=False)
#
#     ############################################################
#     fe_dis = cal_fisher_exact_dis(graph, enriched_SAFE_total, p_value=0.05)
#     import networkx as nx
#
#     edges = []
#     for f1, f2 in itertools.combinations(fe_dis.index, 2):
#         if fe_dis.loc[f1, f2] <= np.percentile(fe_dis.values.reshape(-1,1),0.5):
#             edges.append((f1, f2, {'weight': 1 - fe_dis.loc[f1, f2]}))
#     G = nx.from_edgelist(edges)
#     file_path = '/home/liaoth/Desktop/test.edges'
#     nx.write_edgelist(G, file_path)
#     import pandas as pd
#
#     edge_df = pd.read_csv(file_path, sep=' ', header=None)
#     edge_df.index = range(edge_df.shape[0])
#
#     all_nodes = list(set(list(edge_df.iloc[:, 0]) + list(edge_df.iloc[:, 1])))
#     node_df = pd.DataFrame(index=all_nodes, columns=['cat'])
#     for idx in range(edge_df.shape[0]):
#         source_name = edge_df.iloc[idx, 0]
#         end_name = edge_df.iloc[idx, 1]
#
#         edge_df.loc[idx, 'weight'] = -np.log(fe_dis.loc[source_name, end_name])
#         node_df.loc[source_name, "cat"] = metadata_category.loc[source_name, 'Category']
#         node_df.loc[end_name, "cat"] = metadata_category.loc[end_name, 'Category']
#     node_df.index.name = 'feature'
#     edge_df = edge_df.drop([2, 3], axis=1)
#     edge_df.columns = ['Source', 'End', 'weight']
#     edge_df.to_csv(file_path, index=False, sep='\t')
#     node_df.to_csv(file_path.replace('edge', 'node'), sep='\t')

