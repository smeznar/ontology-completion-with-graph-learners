import sys
sys.path.append("../src")

import time
import networkx as nx
import numpy as np

from models import PredictionModel


class AdamicAdar(PredictionModel):
    def __init__(self):
        self.time = 0
        self.nx_graph = None

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        self.nx_graph = nx.from_scipy_sparse_matrix(adj)
        self.nx_graph.to_undirected()

    def predict(self, test_edges):
        start_time = time.time()
        adar = nx.adamic_adar_index(self.nx_graph, ebunch=test_edges)
        pred = np.array([i[2] for i in adar])

        self.time = time.time() - start_time
        return pred / np.max(pred)
