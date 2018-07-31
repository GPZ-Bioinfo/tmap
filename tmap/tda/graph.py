# pre
from networkx import Graph as G
import os,csv


class Graph(G):
    def __init__(self,data):
        super(Graph, self).__init__()
        self.add_nodes_from(data["nodes"].keys())
        self.add_edges_from(data["edges"])
        self._add_data(data)

        self.graph["params"] = data.get("params","")
        self.samples = list(data.get("sample_names",""))
        # using name of samples instead of idx or samples.
        self.graph["cover_ratio"] = self.cover_ratio(len(data.get("samples")))

    # inner property
    def _add_data(self,data):
        # add sampels
        for n in data["nodes"]:
            self.node[n]["samples"] = list(data["nodes"][n])
        # add postitions
        for n,pos in zip(data["nodes"].keys(),data["node_positions"]):
            self.node[n]["positions"] = list(pos)

    @property
    def params(self):
        return self.graph.get("params", "")

    def cover_ratio(self,n_total):
        n_samples = len(set([sample for _ in self.node for sample in self.node[_]["samples"]]))
        return n_samples/float(n_total) * 100

    # query part
    def query_samples(self,sample):
        # given sample and return list nodes
        all_nodes = [n for n in self.node if sample in self.node[n]["samples"]]
        return all_nodes

    def samples_neighbors(self,sample):
        # given a sample and return list of samples as neighbors
        all_nodes = self.query_samples(sample)
        neighbor_nodes = [n for _ in all_nodes for n in self.neighbors(_)]
        all_samples = []
        for n in neighbor_nodes:
            all_samples += self.node[n]["samples"]
        return all_samples

    def query_intervals(self,node1,node2):
        # given two nodes and return list of samples which shared between nodes.
        samples1 = self.node[node1]['samples']
        samples2 = self.node[node2]['samples']
        return list(set(samples1).intersection(set(samples2)))

    # visualize
    def show(self, data, color=None, fig_size=(10, 10), node_size=10, edge_width=2, mode=None, strength=None):
        pass

    def show_samples(self,samples):
        # give samples and place an obvious color in graph to show them.
        pass

    # output
    def output_nodes(self,filepath,sep='\t'):
        edges = self.edges()
        with open(os.path.realpath(filepath), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=sep)
            spamwriter.writerow(['Source', 'Target'])
            for source, target in edges:
                spamwriter.writerow([source, target])

    def output_pack(self,data):

        pass
        #cache.to_csv('nodes');cache.to_csv('edges')

    # co-enrichment
    def coenrich(self,safe_scores):
        # output another graph
        pass

    #