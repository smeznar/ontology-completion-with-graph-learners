import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/src")

import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from models import Snore
from utils import read_graph_from_json

if __name__ == '__main__':
    num_features = 10

    graph, node_ids, edge_ids = read_graph_from_json("data/go.json")
    adj = nx.to_scipy_sparse_matrix(nx.Graph(graph))

    name_to_node = {node_ids[i]: i for i in range(adj.shape[0])}

    model = Snore()
    model.train(adj, [], [], [], [], None, (0, 0))
    embedding = model.embedding
    feature_names = model.model.selected_features

    feature_values = ((embedding[name_to_node["http://purl.obolibrary.org/obo/GO_0008150"], :]).toarray() *
                      (embedding[name_to_node["http://purl.obolibrary.org/obo/GO_1905288"], :]).toarray())[0]
    sorted_features = np.argsort(-feature_values)

    print("Top features")
    measurements = []
    names = []
    for i in range(num_features):
        names.append(node_ids[feature_names[sorted_features[i]]].split("/")[-1])
        measurements.append(feature_values[sorted_features[i]])
        print("{}, value: {}".format(node_ids[feature_names[sorted_features[i]]], feature_values[sorted_features[i]]))

    measurements = measurements[:4]
    measurements.append(np.mean(feature_values))
    names = names[:4]
    names.append("Mean")

    sns.barplot(x="Feature", y="Value", data=pd.DataFrame(data={"Feature": names, "Value": measurements}))\
        .set_title('Edge {} - {}'.format("GO_0008150", "GO_1905288"))

    plt.show()
