from sklearn.neighbors import NearestNeighbors
import networkx as nx

def density_estimation(graph):
    num_samples = graph.rawX.shape[0]
    data = graph.rawX
    
    k = int(num_samples*0.1)
    neigh = NearestNeighbors(n_neighbors=k, 
                             )
    neigh.fit(data)
    neigh_dist,neigh_ind= neigh.kneighbors(data,
                                           return_distance=True)
    
    KNN_samples = neigh_dist.sum(1)/k

    density_node = {}                           
    for n in graph.nodes:
        samples = graph.node2sample(n)
        num_s = len(samples)
        D_inv_V = KNN_samples[data.index.isin(graph.node2sample(0))].sum()/num_s**2
        density_node[n] = 1/D_inv_V
        graph.nodes[n]['density'] = 1/D_inv_V
    return density_node
    
def states_assignments(graph:nx.Graph,density_node):
    dg = graph.to_directed()
    for n1,n2 in list(dg.edges):
        d1 = density_node[n1]
        d2 = density_node[n2]
        if d1>=d2:
            dg.remove_edge(n1,n2)
    components = nx.attracting_components(dg)
    return components