import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/src")

import networkx as nx
import numpy as np

from models import Snore
from utils import read_graph_from_json

if __name__ == '__main__':
    num_recommendations = 20

    graph, node_ids, edge_ids = read_graph_from_json("../data/go.json")
    adj = nx.to_scipy_sparse_matrix(nx.Graph(graph))

    name_to_node = {node_ids[i]: i for i in range(adj.shape[0])}

    model = Snore()
    model.train(adj, [], [], [], [], None, (0, 0))
    preds = model.predict([(name_to_node['http://purl.obolibrary.org/obo/GO_0008150'], i) for i in range(adj.shape[0])])

    redundant = preds * adj[name_to_node['http://purl.obolibrary.org/obo/GO_0008150'], :].toarray()[0]
    missing = preds - redundant

    sorted_redundant = np.argsort(redundant)
    sorted_missing = np.argsort(-missing)

    i = 0
    print("Redundant")
    for ind in sorted_redundant:
        if adj[name_to_node['http://purl.obolibrary.org/obo/GO_0008150'], ind] > 0.5:
            i += 1
            print("{} - {}, Score: {}".format('http://purl.obolibrary.org/obo/GO_0008150', node_ids[ind], redundant[ind]))
            if i > num_recommendations:
                break
    print()

    print("Missing")
    for i in range(num_recommendations):
        print("{} - {}, Score: {}".format('http://purl.obolibrary.org/obo/GO_0008150', node_ids[sorted_missing[i]], missing[sorted_missing[i]]))


