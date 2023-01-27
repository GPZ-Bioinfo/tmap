from scipy.spatial import distance
import json


def to_json(graph):
    
    mapper_json = {}
    
    adj_matrix = graph.adjmatrix.todense()
    mapper_json['adjacency'] = adj_matrix.tolist()
    mapper_json['num_vertices'] = len(graph.nodes)
    mapper_json['level_of_vertex'] = list(range(1,len(graph.nodes)+1))
    
    points_in_vertex = []
    
    for n in graph.nodes:
        points_in_vertex.append(list(map(int,graph.node2sample(n,rid=True))))

    mapper_json['points_in_vertex'] = points_in_vertex
    mapper_json['points_in_level'] = points_in_vertex
    vertices_in_level = []
    for n in range(1,len(graph.nodes)+1):
        vertices_in_level.append([n])
    
    mapper_json['vertices_in_level'] = vertices_in_level
    
    data = graph.rawX
    with open('./test.json','w') as f1:
        json.dump(mapper_json,f1)
        
    data.T.to_csv('./test_data.csv',index=True,index_label=True)
    
    data.loc[:,['Faecalibacterium','Bacteroides']].to_csv('./test_metadata.csv',index=True,index_label=True)
    
    metadata.to_csv('./test_metadata.csv',index=True,index_label=True)
    