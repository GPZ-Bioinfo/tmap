import networkx as nx


class Graph(nx.Graph):
    """
    Main class of tmap.
    """
    def __init__(self, X, name=''):
        super(Graph, self).__init__(name=name)
        self.rawX = X

    def __repr__(self):
        pass

    def __str__(self):
        pass

    # accessory
    def node2sample(self):
        pass
    def sample2nodes(self):
        pass
    def permute_(self,type='node'):
        pass
    def cal_neighbour(self):
        pass

    # addable
    def add_raw_samples(self):
        pass
    def _recal_dis(self):
        pass
    def update(self):
        pass

    # necessary
    def map(self):
        pass
    def filter(self):
        pass
    def cluster(self):
        pass
    def read(self, filename):
        pass
    def write(self, filename):
        pass
    def quick_view(self):
        pass
    # attr

    @property
    def nodes(self):

        return
    @property
    def sample(self):

        return

    @property
    def cubes(self):

        return
    @property
    def params(self):

        return

    @property
    def status(self):

        return
