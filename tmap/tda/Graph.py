import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from tmap.tda import utils


class Graph(nx.Graph):
    """
    Main class of tmap.
    """

    def __init__(self, X, name=''):
        super(Graph, self).__init__(name=name)
        self.rawX = X  # sample x features
        self.nodePos = None
        self.cal_params = {}
        self.all_spath = None
        self.weight = None

    def info(self):
        description = \
        """
        Graph {name}
        Contains {num_n} nodes and {num_s} samples
        During constructing graph, {loss_p:2f} percent of samples is lost
        
        Used params: 
        {str_p}
        """.format(name=self.name,
                   num_n=len(self.nodes),
                   num_s=len(self.remaining_samples),
                   loss_p=self.cover_ratio(),
                   str_p=self.params
                   )
        description = '\n'.join([_.strip(' ') for _ in description.split('\n')])
        return description

    # accessory
    def check_empty(self):
        if not self.cal_params:
            exit('Graph is empty, please use mapper to create graph instead of directly initiate it.')

    def get_sample_size(self, nodeID):
        n = self.nodes.get(nodeID, {})
        if not n:
            raise nx.NodeNotFound
        return len(n.get('sample', -1))

    def cover_ratio(self):
        self.check_empty()
        return len(self.remaining_samples) / self.rawX.shape[0]

    def node2sample(self, nodeid):
        """
        :param list/str nodeid:
        :return:
        """
        self.check_empty()
        nodes = self.nodes
        samples = []
        if type(nodeid) != int:
            for nid in nodeid:
                samples += list(nodes[nid]['sample'])
        else:
            samples += list(nodes[nodeid]['sample'])
        return list(set(samples))

    def sample2nodes(self, sampleid):
        self.check_empty()
        nodes = self.nodes
        getnodes = []
        if type(sampleid) != int:
            for sid in sampleid:
                getnodes += [nid for nid, attr in nodes.items() if sid in attr['sample']]
        else:
            sid = sampleid
            getnodes += [nid for nid, attr in nodes.items() if sid in attr['sample']]
        return list(set(getnodes))

    def transform_sn(self, data, type='s2n'):
        """
        :param data:
        :param type: s2n mean 'sample to node', n2s mean 'node to sample'
        :return:
        """
        if type == 's2n':
            node_data = utils.transform2node_data(self, data)
            return node_data
        elif type == 'n2s':
            print("From node to sample is a replication process")
            sample_data = utils.transform2sample_data(self, data)
            return sample_data
        else:
            return

    def update_dist(self, weight=None):
        if self.all_spath:
            print("Overwriting existing shortest path and corresponding distant. With assigned weight %s" % (weight if weight else 'default'))
        self.all_spath = {}
        self.all_length = {}
        self.weight = weight
        for n in self:
            # iter node
            self.all_spath[n] = nx.shortest_path(self, n, weight=weight)
            self.all_length[n] = nx.shortest_path_length(self, n, weight=weight)
            # get all pairwise node distance, including self-distance of 0

    def get_neighborhoods(self, nodeid=None, nr_threshold=0.5):
        """
        generate neighborhoods from the graph for all nodes
        :param nr_threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles. nr means neighbour
        :return: return a dict with keys of nodes, values is a list of another node ids.
        """
        all_length = [dist for k1,v1 in self.all_length.items() for dist in v1.values()]
        length_threshold = np.percentile(all_length, nr_threshold)
        if nodeid is not None:
            if type(nodeid) == int:
                nodeid = [nodeid]

        else:
            nodeid = self.nodes
        neighborhoods = {nid: [reach_nid
                               for reach_nid, dis in self.all_length[nid].items()
                               if dis <= length_threshold]
                         for nid in nodeid}
        return neighborhoods

    def neighborhood_score(self, node_data, neighborhoods=None, mode='sum'):
        """
        calculate neighborhood scores for each node from node associated data
        :param node_data: node associated values
        :param _cal_type: hidden parameters. For a big data with too many features(>=100), calculation with pandas will faster than using dict.
        :return: return a dict with keys of center nodes, value is a float
        """
        if neighborhoods is None:
            neighborhoods = self.get_neighborhoods()
        node_data = utils.unify_data(node_data)

        map_fun = {'sum': np.sum,
                   'weighted_sum': np.sum,
                   'weighted_mean': np.mean,
                   "mean": np.mean}
        if mode not in ["sum", "mean", "weighted_sum", "weighted_mean"]:
            raise SyntaxError('Wrong provided parameters.')
        else:
            aggregated_fun = map_fun[mode]

        if 'weighted_' in mode:
            sizes = [self.nodes[nid]['size'] for nid in node_data.index]
            node_data = node_data.multiply(sizes, axis='index')

        # weighted neighborhood scores by node size
        neighborhood_scores = {k: aggregated_fun(node_data.iloc[neighbors,:].values, 0)
                               for k, neighbors in neighborhoods.items()}
        neighborhood_scores = pd.DataFrame.from_dict(neighborhood_scores, orient="index", columns=node_data.columns)
            # neighborhood_scores = neighborhood_scores.reindex(node_data.index)
        return neighborhood_scores


    # addable sample
    def add_raw_samples(self):
        pass

    def _recal_dis(self):
        pass

    def _update(self):
        pass

    # necessary
    def _add_node(self, nodes):
        samples = []
        for nid,attr in nodes:
            samples += list(attr['sample'])
        self.remaining_samples = list(set(samples))
        self.add_nodes_from(nodes)

    def _add_edge(self, edges):
        node_X = self.transform_sn(self.rawX, type='s2n')
        eu_dm = squareform(pdist(node_X, metric='euclidean'))
        self.add_edges_from([(u, v, {'dist': eu_dm[u, v]}) for u, v in edges])
        self.update_dist()

    def _add_node_pos(self, n_pos):
        self.nodePos = n_pos

    def _record_params(self, params):
        """
        {'clusterer': clusterer,
         'cover': cover,
         'lens': self.lens,
         'used_data': {'projected_data': self.projected_data,
                       'filter_data': self.filter_data}}

        :param params:
        :return:
        """
        self.cal_params.update(params)

    def read(self, filename):
        pass

    def write(self, filename):
        pass

    def quick_view(self):
        pass

    # attr

    @property
    def adjmatrix(self):
        return nx.adj_matrix(self)

    @property
    def cubes(self):
        self.check_empty()
        cover = self.cal_params['cover']
        cubes = cover.hypercubes
        return cubes

    @property
    def params(self):
        template_text = \
        """
        cluster params
        {cluster_p}
        =================
        cover params
        {cover_p}
        =================
        lens params
        {lens_p}
        """
        p = self.cal_params
        cluster_p = p['clusterer'].get_params()
        cover_p = {'r': p['cover'].resolution,
                   'overlap': p['cover'].overlap}
        lens_p = {'lens_%s' % idx: {'components': len.components,
                                    'metric': 'none' if len.metric is None else len.metric.name}
                  for idx, len in enumerate(p['lens'])}
        params = template_text.format(
            cluster_p='\n'.join(['%s: %s' % (k,
                                               v)
                                 for k, v in cluster_p.items()]),
            cover_p='\n'.join(['%s: %s' % (k,
                                             v)
                               for k, v in cover_p.items()]),
            lens_p='\n'.join(['%s:\n%s' % (k,
                                             '\n'.join(['%s: %s' % (_k,_v)
                                                       for _k,_v in v.items()])) for k, v in lens_p.items()])
        )
        params = '\n'.join([_.strip(' ') for _ in params.split('\n')])
        return params

    @property
    def status(self):
        return
