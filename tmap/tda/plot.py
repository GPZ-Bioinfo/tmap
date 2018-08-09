# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import colorsys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
from sklearn import decomposition


class Color(object):
    """
    map colors to target values for TDA network visualization
    """
    def __init__(self, target, dtype="numerical", target_by="sample"):
        """
        :param target: target values for samples or nodes
        :param dtype: type of target values, "numerical" or "categorical"
        :param target_by: target type of "sample" or "node"
        (for node target values, accept a node associated dictionary of values)
        """
        if target is None:
            raise Exception("target must not be None.")

        # for node target values, accept a node associated dictionary of values
        if target_by == 'node':
            _target = np.zeros(len(target))
            for _node_idx, _node_val in target.items():
                _target[_node_idx] = _node_val
            target = _target

        if type(target) is not np.ndarray:
            target = np.array(target)
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)
        if dtype not in ["numerical", "categorical"]:
            raise ValueError("data type must be 'numerical' or 'categorical'.")
        if target_by not in ["sample", "node"]:
            raise ValueError("target values must be by 'sample' or 'node'")
        # target values should be numbers, check and encode categorical labels

        if ((type(target[0][0]) != int)
                and (type(target[0][0]) != float)
                and (not isinstance(target[0][0],np.number))
        ):
            self.label_encoder = LabelEncoder()
            self.target = self.label_encoder.fit_transform(target)
        else:
            self.label_encoder = None
            self.target = target

        self.dtype = dtype
        self.labels = target
        self.target_by = target_by

    def _get_hex_color(self, i):
        """
        map a normalize i value to HSV colors
        :param i: input for the hue value, normalized to [0, 1.0]
        :return: a hex color code for i
        """
        # H values: from 0 (red) to 240 (blue), using the HSV color systems for color mapping
        # largest value of 1 mapped to red, and smallest of 0 mapped to blue
        c = colorsys.hsv_to_rgb((1 - i) * 240 / 360, 1.0, 0.75)
        return "#%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

    def _rescale_target(self, target):
        """
        scale target values according to density/percentile
        to make colors distributing evenly among values
        :param target: numerical target values
        :return:
        """
        rescaled_target = np.zeros(target.shape)

        scaler_min_q1 = MinMaxScaler(feature_range=(0, 0.25))
        scaler_q1_median = MinMaxScaler(feature_range=(0.25, 0.5))
        scaler_median_q3 = MinMaxScaler(feature_range=(0.5, 0.75))
        scaler_q3_max = MinMaxScaler(feature_range=(0.75, 1))

        q1, median, q3 = np.percentile(target, 25), np.percentile(target, 50), np.percentile(target, 75)

        # if len(set([q1,median,q3])) != 3:
        #     same_bounds = []
        index_min_q1 = np.where(target <= q1)[0]
        index_q1_median = np.where(((target >= q1) & (target <= median)))[0]
        index_median_q3 = np.where(((target >= median) & (target <= q3)))[0]
        index_q3_max = np.where(target >= q3)[0]

        target_min_q1 = scaler_min_q1.fit_transform(target[index_min_q1])
        target_q1_median = scaler_q1_median.fit_transform(target[index_q1_median])
        target_median_q3 = scaler_median_q3.fit_transform(target[index_median_q3])
        target_q3_max = scaler_q3_max.fit_transform(target[index_q3_max])

        if all(target_q3_max == 0.75):
            target_q3_max = np.ones(target_q3_max.shape)
        if q1 == median == q3:
            target_q3_max = np.array([_ if _!= 0.75 else 0 for _ in target_q3_max[:,0]]).reshape(target_q3_max.shape)
        rescaled_target[index_median_q3] = target_median_q3
        rescaled_target[index_q1_median] = target_q1_median
        rescaled_target[index_min_q1] = target_min_q1
        rescaled_target[index_q3_max] = target_q3_max

        return rescaled_target

    def get_colors(self, nodes, cmap=None):
        """
        :param nodes:
        :param cmap: not implemented yet...
        :return: nodes colors with keys, and the color map of the target values
        """
        # todo: accept a customzied color map [via the 'cmap' parameter]
        node_keys = nodes.keys()

        # map a color for each node
        node_color_idx = np.zeros((len(nodes), 1))
        for i, node_id in enumerate(node_keys):
            if self.target_by == 'node':
                target_in_node = self.target[node_id]
            else:
                target_in_node = self.target[nodes[node_id]]

            # summarize target values from samples/nodes for each node
            if self.dtype == "categorical":
                # most common value (if more than one, the smallest is return)
                node_color_idx[i] = stats.mode(target_in_node)[0][0]
            elif self.dtype == "numerical":
                node_color_idx[i] = np.mean(target_in_node)

        _node_color_idx = self._rescale_target(node_color_idx)
        node_colors = [self._get_hex_color(idx) for idx in _node_color_idx]

        return dict(zip(node_keys, node_colors)), (node_color_idx, node_colors)


def show(data, graph, color=None, fig_size=(10, 10), node_size=10, edge_width=2, mode=None, strength=None):
    """
    network visualization of TDA mapper
    :param data:
    :param graph:
    :param color:
    :param fig_size:
    :param node_size:
    :param edge_width:
    :param mode:
    :param strength:
    :return:
    """
    # todo: add file path for graph saving
    node_keys = graph["node_keys"]
    node_positions = graph["node_positions"]
    node_sizes = graph["node_sizes"]

    # scale node sizes
    max_node_size = np.max(node_sizes)
    sizes = (node_sizes / max_node_size) * (node_size ** 2)

    # map node colors
    if color is None or type(color) == str:
        if color is None:
            color = 'red'
        color_map = {node_id: color for node_id in node_keys}
        target2colors = (np.zeros((len(node_keys), 1)),[color] * len(node_keys))
    else:
        color_map, target2colors = color.get_colors(graph["nodes"])
    colorlist = [color_map[it] for it in node_keys]

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1)

    node_target_values, node_colors = target2colors
    legend_lookup = dict(zip(node_target_values.reshape(-1,), node_colors))

    # add categorical legend
    if isinstance(color,Color):
        if color.dtype == "categorical":
            for label in set([it[0] for it in color.labels]):
                if color.label_encoder:
                    label_color = legend_lookup[color.label_encoder.transform([label])[0]]
                else:
                    label_color = legend_lookup[label]
                ax.plot([], [], 'o', color=label_color, label=label, markersize=10)
            legend = ax.legend(numpoints=1, loc="upper right")
            legend.get_frame().set_facecolor('#bebebe')

        # add numerical colorbar
        elif color.dtype == "numerical":
            legend_values = sorted([_ for _ in legend_lookup])
            legned_colors = [legend_lookup[_] for _ in legend_values]

            cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', legned_colors)
            norm = mcolors.Normalize(min(legend_values), max(legend_values))
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cb = fig.colorbar(sm, shrink=0.5)
            cb.ax.yaxis.set_ticks_position('right')
            cb.ax.text(0.5, -0.02, '%.2f' % min(legend_values), ha='center', va='top', weight='bold')
            cb.ax.text(0.5, 1.02, '%.2f' % max(legend_values), ha='center', va='bottom', weight='bold')

    if mode == 'spring':
        pos = {}
        # the projection space is one dimensional
        if node_positions.shape[1] == 1:
            m = decomposition.PCA(n_components=2)
            s = MinMaxScaler()
            d = m.fit_transform(data)
            d = s.fit_transform(d)
            for k in node_keys:
                data_in_node = d[graph['nodes'][k]]
                pos.update({int(k): np.average(data_in_node, axis=0)})
        elif node_positions.shape[1] >= 2:
            for i, k in enumerate(node_keys):
                pos.update({int(k): node_positions[i, :2]})

        G = nx.Graph(pos=pos)
        G.add_nodes_from(node_keys)
        G.add_edges_from(graph["edges"])
        pos = nx.spring_layout(G, pos=pos, k=strength)
        # add legend
        nx.draw_networkx(G, pos=pos, node_size=sizes,
                         node_color=colorlist,
                         width=edge_width,
                         edge_color=[color_map[edge[0]] for edge in graph["edges"]],
                         with_labels=False, label="0", ax=ax)
    else:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        node_idx = dict(zip(node_keys, range(len(node_keys))))
        for edge in graph["edges"]:
            ax.plot([node_positions[node_idx[edge[0]], 0], node_positions[node_idx[edge[1]], 0]],
                    [node_positions[node_idx[edge[0]], 1], node_positions[node_idx[edge[1]], 1]],
                    c=color_map[edge[0]], zorder=1)
        ax.scatter(node_positions[:, 0], node_positions[:, 1],
                   c=colorlist, s=sizes, zorder=2)

    plt.axis("off")
    plt.show()
