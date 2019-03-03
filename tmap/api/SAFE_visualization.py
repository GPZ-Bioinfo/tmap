import plotly
from plotly import tools
from plotly import graph_objs as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def draw_PCOA():
    pass

def draw_ranking():
    fig = tools.make_subplots(1, 4, shared_yaxes=True, horizontal_spacing=0, subplot_titles=['Kendalltau test',
                                                                                             'Ranking based SAFE enriched score',
                                                                                             'Ranking reported envfit R',
                                                                                             'Ranking envfit R'])

    sorted_df = safe_summary_metadata.sort_values('SAFE enriched score', ascending=False)
    fig.append_trace(go.Scatter(x=compared_table.loc[:, 'kendalltau_test_correlation'],
                                y=compared_table.index,
                                mode='markers+lines',
                                marker=dict(color=[1 if _ <= 0.05 else 0 for _ in
                                                   compared_table.loc[:, 'kendalltau_test_pvalue']],
                                            ),
                                showlegend=False), 1, 1)
    fig.append_trace(go.Bar(x=compared_table.loc[:, 'SAFE enriched score'],
                            y=compared_table.index,
                            marker=dict(
                                color=[color_codes[metadata_category.loc[fea, 'Category']] for fea in sorted_df.index],
                                line=dict(width=1)),
                            orientation='h',
                            showlegend=False), 1, 2)
    fig.append_trace(go.Bar(x=compared_table.loc[:, 'original R'],
                            y=compared_table.index,
                            orientation='h',
                            showlegend=False), 1, 3)
    fig.append_trace(go.Bar(x=compared_table.loc[:, 'R'],
                            y=compared_table.index,
                            orientation='h',
                            showlegend=False), 1, 4)

    fig.layout.yaxis.autorange = 'reversed'
    fig.layout.margin.l = 400
    fig.layout.height = 1500
    plotly.offline.iplot(fig)  # filename='fig4a.compared Bar plot.html',auto_open=False)
    plotly.offline.plot(fig, filename='result/compared Bar plot.html', auto_open=False)
    bar_plot2_table = pd.DataFrame(index=sorted_df.index, columns=['Ranking based SAFE enriched score',
                                                                   'Ranking reported envfit R',
                                                                   'Ranking envfit R'])
    bar_plot2_table.loc[:, 'Ranking based SAFE enriched score'] = sorted_df.loc[:, 'SAFE enriched score']
    bar_plot2_table.loc[:, 'Ranking reported envfit R'] = input_otu.apply(lambda x: len(x[x != 0]) / len(x), axis=0)[
        sorted_df.index]
    bar_plot2_table.loc[:, 'Ranking envfit R'] = input_otu.div(input_otu.sum(1), axis=0).sum(0)[sorted_df.index]
    compared_table.to_csv("result/Bar raw data.csv")

def draw_network():
    pass