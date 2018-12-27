fea = [_ for _ in emp_metadata.columns if _.startswith('empo_2')][0]  # randomly choose one of them
global_correlative_feas, sub_correlative_feas, metainfo = coenrichment_for_nodes(graph,
                                                                                 enriched_centroides[fea],
                                                                                 fea,
                                                                                 enriched_centroides,
                                                                                 _filter=True,
                                                                                 mode='both')
global_corr_df, sub_corr_df = construct_correlative_metadata(fea, global_correlative_feas, sub_correlative_feas, metainfo, pd.concat([node_data,
                                                                                                                                      nodes_metadata], axis=1), verbose=0)
