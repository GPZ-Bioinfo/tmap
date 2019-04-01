#! /usr/bin/python3
import argparse
import pickle
from collections import Counter

import plotly

from plotly import graph_objs as go
from plotly import tools
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from tmap.api.general import *
from tmap.tda.plot import vis_progressX, Color
from tmap.tda.Graph import Graph
from tmap.tda.utils import output_fig


def draw_PCOA(rawdatas, summary_datas, output, mode='html', width=1500, height=1000, sort_col='SAFE enriched score'):
    """
    Currently ordination visualization just support pcoa.
    :param rawdata:
    :param summary_data:
    :param output:
    :param mode:
    :param width:
    :param height:
    :param sort_col:
    :return:
    """
    fig = go.Figure()
    summary_data = pd.concat(summary_datas, axis=0)
    # it won't raise error even it only contains one df.
    safe_dfs = [pd.DataFrame.from_dict(r_dict) for r_dict in rawdatas]  # row represents nodes, columns represents features.
    safe_df = pd.concat(safe_dfs, axis=1)
    safe_df = safe_df.reindex(columns=summary_data.index)

    pca = PCA()
    pca_result = pca.fit_transform(safe_df.T)

    mx_scale = MinMaxScaler(feature_range=(10, 40)).fit(summary_data.loc[:, [sort_col]])
    top10_feas = list(summary_data.sort_values(sort_col, ascending=False).index[:10])

    for each in summary_datas:
        vals = each.loc[:, [sort_col]]
        fig.add_trace(go.Scatter(x=pca_result[safe_df.columns.isin(each.index), 0],
                                 y=pca_result[safe_df.columns.isin(each.index), 1],
                                 mode="markers",
                                 marker=dict(  # color=color_codes[cat],
                                     size=mx_scale.transform(vals),
                                     opacity=0.5),
                                 showlegend=True if len(summary_datas) > 1 else False,
                                 text=safe_df.columns[safe_df.columns.isin(each.index)]))

    fig.add_trace(go.Scatter(x=pca_result[safe_df.columns.isin(top10_feas), 0],
                             y=pca_result[safe_df.columns.isin(top10_feas), 1],
                             # visible=False,
                             mode="text",
                             hoverinfo='none',
                             textposition="middle center",
                             name='name for searching',
                             showlegend=False,
                             textfont=dict(size=13),
                             text=top10_feas))

    fig.layout.update(dict(xaxis=dict(title="PC1({:.2f}%)".format(pca.explained_variance_ratio_[0] * 100)),
                           yaxis=dict(title="PC2({:.2f}%)".format(pca.explained_variance_ratio_[1] * 100)),
                           width=width,
                           height=height,
                           font=dict(size=15),
                           hovermode='closest', ))

    output_fig(fig,output,mode)
    logger("Ordination graph has been output to", output, verbose=1)


def draw_stratification(graph, SAFE_dict, cols, output, mode='html', n_iter=1000, p_val=0.05, width=1000, height=1000, allnodes=False):
    # Enterotyping-like stratification map based on SAFE score

    node_pos = graph.nodePos
    sizes = graph.size
    nodes = graph.nodes
    sizes = np.array([sizes[_] for _ in range(len(nodes))]).reshape(-1, 1)

    transformed_sizes = MinMaxScaler(feature_range=(10, 40)).fit_transform(sizes).ravel()
    xs = []
    ys = []
    for edge in graph.edges:
        xs += [node_pos[edge[0], 0],
               node_pos[edge[1], 0],
               None]
        ys += [node_pos[edge[0], 1],
               node_pos[edge[1], 1],
               None]
    fig = plotly.tools.make_subplots(1, 1)

    node_line = go.Scatter(
        # ordination line
        visible=True,
        x=xs,
        y=ys,
        marker=dict(color="#8E9DA2",
                    opacity=0.7),
        line=dict(width=1),
        showlegend=False,
        hoverinfo='skip',
        mode="lines")
    fig.append_trace(node_line, 1, 1)

    safe_score_df = pd.DataFrame.from_dict(SAFE_dict)  # row: nodes, columns: features
    min_p_value = 1.0 / (n_iter + 1.0)
    SAFE_pvalue = np.log10(p_val) / np.log10(min_p_value)
    tmp = [safe_score_df.columns[_] if safe_score_df.iloc[idx, _] >= SAFE_pvalue else np.nan for idx, _ in enumerate(np.argmax(safe_score_df.values, axis=1))]
    # get enriched features with biggest SAFE_score per nodes.
    t = Counter(tmp)
    # number of (imp) features among all nodes. (imp: with biggest SAFE score per node compared other features at same node and bigger than p_val)
    if cols:
        if any([_ not in safe_score_df.columns for _ in cols]):
            logger("There are provided cols \" %s\"doesn't at SAFE summary table." % ';'.join(cols), verbose=1)
        for fea in cols:
            if allnodes:
                color = Color(SAFE_dict[fea], target_by='node', dtype='numerical')
                subfig = vis_progressX(graph,
                                       simple=True,
                                       mode='obj',
                                       color=color
                                       )
                subfig.data[1]['name'] = fea
                fig.append_trace(subfig.data[1], 1, 1)
            else:
                get_nodes_bool = (safe_score_df.loc[:, fea] >= SAFE_pvalue).all()
                if not get_nodes_bool:
                    # if all False....
                    logger("fea: %s get all False bool indicated there are not enriched nodes showed at the graph" % fea, verbose=1)
                else:
                    node_position = go.Scatter(
                        # node position
                        visible=True,
                        x=node_pos[get_nodes_bool, 0],
                        y=node_pos[get_nodes_bool, 1],
                        hoverinfo="text",
                        marker=dict(  # color=node_colors,
                            size=[sizes[_,0] for _ in np.arange(node_pos.shape[0])[get_nodes_bool]],
                            opacity=0.9),
                        showlegend=True,
                        name=str(fea) + ' (%s)' % str(t.get(fea, 0)),
                        mode="markers")
                    fig.append_trace(node_position, 1, 1)
    else:
        for idx, fea in enumerate([_ for _, v in sorted(t.items(), key=lambda x: x[1]) if v >= 10]):
            # safe higher than threshold, just centroides
            node_position = go.Scatter(
                # node position
                visible=True,
                x=node_pos[np.array(tmp) == fea, 0],
                y=node_pos[np.array(tmp) == fea, 1],
                hoverinfo="text",
                marker=dict(  # color=node_colors,
                    size=[transformed_sizes[_] for _ in np.arange(node_pos.shape[0])[np.array(tmp) == fea]],
                    opacity=0.9),
                showlegend=True,
                name=str(fea) + ' (%s)' % str(t[fea]),
                mode="markers")
            fig.append_trace(node_position, 1, 1)
    fig.layout.width = width
    fig.layout.height = height
    fig.layout.font.size = 15
    fig.layout.hovermode = 'closest'

    output_fig(fig,output,mode)
    logger("Stratification graph has been output to", output, verbose=1)


def process_summary_paths(safe_summaries):
    datas = [data_parser(path, verbose=0) for path in safe_summaries]
    if len(datas) > 1:
        cols_dict = {}
        for path, data in zip(safe_summaries, datas):
            name = os.path.basename(path).strip('.csv')
            data.columns = ['%s (%s)' % (col, name) for col in data.columns]
            cols_dict[name] = list(data.columns)
        data = pd.concat(datas, axis=0)
    else:
        data = datas[0]
        cols_dict = {'Only one df': data.columns}
    return data, cols_dict


def draw_ranking(data, cols_dict, output, mode='html', width=1600, height=1400, sort_col='SAFE enriched score'):
    col_names = list(cols_dict.keys())
    if len(col_names) == 1:
        fig = tools.make_subplots(1, 1)
    else:
        fig = tools.make_subplots(1, len(cols_dict), shared_yaxes=True, horizontal_spacing=0, subplot_titles=col_names)

    sorted_cols = [_ for _ in data.columns if _.startswith(sort_col)]
    if not sorted_cols:
        logger("data you provide doesn't contain columns like %s, Maybe you provide a metadata directly? instead of SAFE summary table." % sort_col, verbose=1)
        sorted_df = data
    else:
        sorted_df = data.sort_values([_ for _ in data.columns if _.startswith(sort_col)], ascending=False)

    def _add_trace(name, col):
        fig.append_trace(go.Bar(x=sorted_df.loc[:, name],
                                y=sorted_df.index,
                                marker=dict(
                                    line=dict(width=1)
                                ),
                                orientation='h',
                                showlegend=False),
                         1, col)

    for idx, each in enumerate(col_names):
        col = idx + 1

        name = [_ for _ in cols_dict[each] if _.startswith(sort_col)]
        if not name and [_ for _ in cols_dict[each] if _.startswith('r2')]:
            name = [_ for _ in cols_dict[each] if _.startswith('r2')][0]
        elif name:
            name = name[0]
        else:
            logger("Unkown input file.", verbose=1)
        _add_trace(name, col)

    fig.layout.yaxis.autorange = 'reversed'

    fig.layout.margin.l = width / 4
    fig.layout.width = width
    fig.layout.height = height

    output_fig(fig,output,mode)
    logger("Ranking graph has been output to", output, verbose=1)


def main(args):
    if args.mission.lower() == 'ranking':
        data, cols_dict = process_summary_paths(args.sum_s)
        draw_ranking(data=data,
                     cols_dict=cols_dict,
                     output=args.output,
                     mode=args.type,
                     height=args.height,
                     width=args.width,
                     sort_col=args.sort)
    elif args.mission.lower() == 'stratification':
        dict_data = pickle.load(open(args.SAFE[0], 'rb'))
        safe_dict = dict_data['data']
        n_iter = dict_data['params']['n_iter']
        graph = Graph().read(args.graph)
        draw_stratification(graph=graph,
                            SAFE_dict=safe_dict,
                            cols=args.col,
                            output=args.output,
                            mode=args.type,
                            n_iter=n_iter,
                            p_val=args.pvalue,
                            width=args.width,
                            height=args.height,
                            allnodes=args.allnodes)
    elif args.mission.lower() == 'ordination':
        dict_datas = [pickle.load(open(rawSAFE, 'rb')) for rawSAFE in args.SAFE]
        safe_dicts = [dict_data['data'] for dict_data in dict_datas]
        summary_datas = [data_parser(path, verbose=0) for path in args.sum_s]

        if len(summary_datas) != len(safe_dicts):
            logger("Warning!!! The number of raw data didn't equal to the number of summary datas. It may occurs error.", verbose=1)

        draw_PCOA(rawdatas=safe_dicts,
                  summary_datas=summary_datas,
                  output=args.output,
                  mode=args.type,
                  height=args.height,
                  width=args.width,
                  sort_col=args.sort)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mission", help="Which kinds of graph you want to generate. \
                         [ranking|stratification|ordination]",
                        type=str, choices=['ranking', 'stratification', 'ordination'])
    parser.add_argument("-G", "--graph", help="Graph file computed from 'Network_generator.py'.",
                        type=str)
    parser.add_argument("-O", "--output", help="Prefix of output file",
                        type=str)
    parser.add_argument("-S1", "--SAFE", nargs='*', help="Pickled dict contains raw SAFE scores.",
                        type=str)
    parser.add_argument("-S2", "--SAFE_summary", dest='sum_s', nargs='*', help="Summary of SAFE scores",
                        type=str)
    parser.add_argument("--col", nargs='*', help="The features of metadata you want to focus. (could be multiple.) Only useful for stratification")
    parser.add_argument("--sort", help="The column you need to sort with",
                        type=str, default='SAFE enriched score')
    parser.add_argument("-p", "--pvalue",
                        help="p-val for decide which level of data should consider as significant",
                        default=0.05, type=float)
    parser.add_argument("--type", help="The file type to output figure. [pdf|html|png]",
                        type=str, default='html')
    parser.add_argument("--width", help="The height of output picture",
                        type=int, default=1600)
    parser.add_argument("--height", help="The width of output picture",
                        type=int, default=1600)
    parser.add_argument("--allnodes", help="draw all nodes with provided columns as color instead of enriched one. \nOnly useful for stratification",
                        action="store_true")
    args = parser.parse_args()

    process_output(output=args.output)

    main(args)
