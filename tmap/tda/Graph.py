import itertools,pickle

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist

from tmap.tda import utils
from tmap.tda.plot import show, Color


class Graph(nx.Graph):
    """
    Main class of tmap.
    """

    def __init__(self, X=None, name=''):
        super(Graph, self).__init__(name=name)
        X = utils.unify_data(X)
        self.rawX = X  # sample x features
        self.nodePos = None
        self.cal_params = {}
        self.all_spath = None
        self.weight = None
        self._SAFE = []

    def info(self):
        description = \
            """
            Graph {name}
            Contains {num_n} nodes and {num_s} samples
            During constructing graph, {loss_n} ({loss_p}%) samples lost
            
            Used params: 
            {str_p}
            """.format(name=self.name,
                       num_n=len(self.nodes),
                       num_s=len(self.remaining_samples),
                       loss_n=self.rawX.shape[0] - len(self.remaining_samples),
                       loss_p=round(self.cover_ratio(), 4) * 100,
                       str_p=self.params
                       )
        description = '\n'.join([_.strip(' ') for _ in description.split('\n')])
        return description

    def __repr__(self):
        description = """
            Graph {name}
            Contains {num_n} nodes and {num_s} samples
            """.format(name=self.name,
                       num_n=len(self.nodes),
                       num_s=len(self.remaining_samples))
        return description

    # accessory function
    ## check or is or confirm
    def check_empty(self):
        if not self.cal_params:
            exit('Graph is empty, please use mapper to create graph instead of directly initiate it.')

    def is_sample_name(self, sname):
        if type(sname) == str or type(sname) == int:
            if sname not in self.rawX.index:
                return False
        else:
            for sn in sname:
                if sn not in self.rawX.index:
                    return False
        return True

    def is_samples_shared(self, sample):
        """
        :param sample: name or index of a sample
        :return:
        """
        existnodes = self.sample2nodes(sample)
        if existnodes is None:
            return False
        possible_edges = itertools.combinations(existnodes, 2)
        for e in possible_edges:
            if e in self.edges:
                return True
        return False

    def is_samples_dropped(self, sample):
        """
        :param sample: name or index of a sample
        :return:
        """
        if sample in self.get_dropped_samples() or \
                sample in [self.sname2sid(dsample) for dsample
                           in self.get_dropped_samples()]:
            return True
        else:
            return False

    ## query
    def get_sample_size(self, nodeID):
        n = self.nodes.get(nodeID, {})
        if not n:
            raise nx.NodeNotFound
        return len(n.get('sample', -1))

    def cover_ratio(self):
        self.check_empty()
        return len(self.remaining_samples) / self.rawX.shape[0]

    def samples_neighbors(self, sample_name, nr_threshold=0.5):
        """
        provide single, if dropped samples, will print error message.
        provide multiple samples, if one of them iss dropped, it won't print error message. Please be careful by yourself.
        :param sample_name: name or index of samples, could be multiple or single
        :param nr_threshold:
        :return:
        """
        getnodes = self.sample2nodes(sample_name)
        if getnodes is None:
            # must stop, else neighborhoods will return all neighborhoods
            return []
        getneighborhoods = self.get_neighborhoods(getnodes, nr_threshold=nr_threshold)
        neighbor_samples = self.sample2nodes(getneighborhoods)
        neighbor_sample_names = self.sid2sname(neighbor_samples)
        return neighbor_sample_names

    def get_component_nodes(self, nodes):
        """
        Given a list of enriched nodes which comes from ``get_enriched_nodes``. Normally it the enriched_centroid instead of the others nodes around them.
        :param list enriched_nodes: list of nodes ID which is enriched with given feature and threshold.
        :return: A nested list which contain multiple list of nodes which is not connected to each other.
        """
        g = nx.subgraph(self, nodes)
        comp_nodes = [nodes for nodes in nx.algorithms.components.connected_components(g)]
        return comp_nodes

    def get_shared_samples(self, node_u, node_v):
        """
        :param node_u: name  of node
        :param node_v: name  of node
        :return:
        """
        if self.get_edge_data(node_u, node_v) is not None:
            s1, s2 = self.node2sample(node_u), self.node2sample(node_v)
            shared_sample_names = set(s1).intersection(set(s2))
            return shared_sample_names

    def get_dropped_samples(self):
        if self.remaining_samples:
            dropped_sample_ids = set(np.arange(self.rawX.shape[0])).difference(set(self.remaining_samples))
            dropped_sample_names = [self.sid2sname(int(sid)) for sid in dropped_sample_ids]
            return dropped_sample_names
        else:
            print('No samples remained because of invalid graph construction or no samples clustering.')

    ## indirect attr (For SAFE calculation)
    def get_neighborhoods(self, nodeid=None, nr_threshold=0.5, nr_dist=None):
        """
        generate neighborhoods from the graph for all nodes
        :param nr_threshold: Float in range of [0,100]. The threshold is used to cut path distance with percentiles. nr means neighbour
        :return: return a dict with keys of nodes, values is a list of another node ids.
        """
        all_length = [dist for k1, v1 in self.all_length.items() for dist in v1.values()]
        all_length = [_ for _ in all_length if _ > 0]
        # remove self-distance (that is 0)
        if nr_dist is None:
            length_threshold = np.percentile(all_length, nr_threshold)
        else:
            length_threshold = nr_dist

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

        nv = node_data.values
        # weighted neighborhood scores by node size
        neighborhood_scores = {k: aggregated_fun(nv[neighbors, :], 0)
                               for k, neighbors in neighborhoods.items()}
        neighborhood_scores = pd.DataFrame.from_dict(neighborhood_scores,
                                                     orient="index",
                                                     columns=node_data.columns)
        # neighborhood_scores = neighborhood_scores.reindex(node_data.index)
        return neighborhood_scores

    ## convertor
    def sid2sname(self, sid):
        """
        :param sid:
        :return:
        :rtype list
        """
        # convert sample id into sample name.
        self.check_empty()
        if type(sid) == int:
            sids = [sid]
        else:
            sids = sid

        if not sids:
            # if provide empty sids.
            return ''

        r = self.rawX.index[np.array(sids)]
        if len(r) > 1:
            return list(r)
        elif len(r) == 1:
            return r[0]
        else:
            return r

    def sname2sid(self, sname):
        # sname must be single
        self.check_empty()
        if sname not in self.rawX.index:
            print('Error because of searched sample name not in raw data X')
        else:
            return self.rawX.index.get_loc(sname)

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

        return self.sid2sname(list(set(samples)))

    def sample2nodes(self, sampleid):
        """

        :param sampleid: multiple/single index/name of samples.
        :return:
        """
        self.check_empty()

        nodes = self.nodes
        getnodes = []

        if type(sampleid) == int or type(sampleid) == str:
            sampleid = [sampleid]
        elif type(sampleid) == dict:
            # dict like {nid: [sampleid1,sampleid2...]...}
            sampleid = set([v for k, d in sampleid.items() for v in d])
        else:
            pass
        # process multi id provided situation.

        for sid in sampleid:
            if self.is_sample_name(sid):
                # if sample name provide , return sample index.
                sid = self.sname2sid(sid)
            getnodes += [nid for nid, attr in nodes.items() if sid in attr['sample']]

        if len(set(getnodes)) >= 1:
            return list(set(getnodes))
        else:
            print("Maybe dropped sample provided.....")
            return

    def transform_sn(self, data, type='s2n'):
        """
        s2n mean sample2node, normally, it just transform the data according to the graph.nodes.
        n2s mean node2sample. it actually can't achieve, so it just a process duplicating nodes into samples shape.
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

    ## update function
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

    # for addable and reuseable (todo)
    def add_raw_samples(self):
        pass

    def _recal_dis(self):
        pass

    def _update(self):
        pass

    # necessary
    def _add_node(self, nodes):
        samples = []
        for nid, attr in nodes:
            samples += list(attr['sample'])
        self.remaining_samples = list(set(samples))
        # it must be index of sample instead of sample name
        self.add_nodes_from(nodes)

    def _add_edge(self, edges):
        node_X = self.transform_sn(self.rawX, type='s2n')
        eu_dm = squareform(pdist(node_X, metric='euclidean'))
        self.add_edges_from([(u, v, {'dist': eu_dm[u, v]}) for u, v in edges])
        self.update_dist()

    def _add_node_pos(self, n_pos):
        self.nodePos = n_pos  # average pos from cover.data

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

    def _add_safe(self, params):
        self._SAFE.append(params)

    def clear_safe(self, force=False):
        if not force:
            t = input("Make sure you want to clear all SAFE scores store in graph. Y/y")
            if t.lower() == 'y':
                self._SAFE = []
        else:
            self._SAFE = []
        print('SAFE scores has been cleared.')

    def _add_other_node_attr(self, node_dict=None, suffix=''):
        # for add SAFE into graph
        pass

    # IO part
    def read(self, filename):
        g = pickle.load(open(filename,'rb'))
        return g

    def write(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    # visualization
    def show(self, **kwargs):
        if 'mode' not in kwargs:
            show(self, mode=None, **kwargs)
        else:
            show(self, **kwargs)

    def show_samples(self, samples, **kwargs):
        nids = self.sample2nodes(samples)
        target = [1 if nid in nids else 0 for nid in self.nodes]
        color = Color(target, target_by='node', dtype='categorical')
        show(self, mode=None, color=color, **kwargs)

    # attr
    @property
    def size(self):
        self.check_empty()
        return {n: self.nodes[n]['size'] for n in self.nodes}

    @property
    def sample_names(self):
        self.check_empty()
        return self.rawX.index

    @property
    def data(self):
        """
        projected data which passed to ``filter``
        :return:
        """
        return self.cal_params['used_data']['projected_data']

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
                                           '\n'.join(['%s: %s' % (_k, _v)
                                                      for _k, _v in v.items()])) for k, v in lens_p.items()])
        )
        params = '\n'.join([_.strip(' ') for _ in params.split('\n')])
        return params

    @property
    def status(self):
        return
