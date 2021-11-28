import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/src")

import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp

from models import Snore
from utils import read_graph_from_json
from link_prediction import methods, create_splits


def create_embedding_matrix(emb, pos_edges, neg_edges):
    embeddings = []
    for i, j, k in (pos_edges + neg_edges):
        embeddings.append(sp.csr_matrix.multiply(emb[i], emb[j]))
    return sp.vstack(embeddings), np.array([1 for i in range(len(pos_edges))] + [0 for i in range(len(neg_edges))])


if __name__ == '__main__':
    num_features = 5

    graph, node_ids, edge_ids = read_graph_from_json("data/go.json")
    splits = create_splits(graph)
    adj = nx.to_scipy_sparse_matrix(nx.Graph(graph))

    model = Snore()
    model.train(adj, [], [], [], [], None, (0, 0))
    embedding = model.embedding
    feature_names = model.model.selected_features

    train_mat, train_labels = create_embedding_matrix(embedding, splits[0][1] + splits[0][3] + splits[0][5],
                                                      splits[0][2] + splits[0][4] + splits[0][6])

    edge_classifier = LogisticRegression(random_state=0, solver='liblinear')
    edge_classifier.fit(train_mat, train_labels)

    predProbs = edge_classifier.predict_proba(train_mat)
    X_design = sp.hstack([np.ones((train_mat.shape[0], 1)), train_mat])
    V = np.product(predProbs, axis=1)

    covLogit = np.linalg.pinv(sp.csr_matrix.multiply(X_design.T, V).dot(X_design).toarray())
    se = np.sqrt(np.diag(covLogit))[1:]
    coefs = np.abs(edge_classifier.coef_ / se)[0]
    sorted_coefs = np.argsort(-coefs)

    relevant_parameters = sorted_coefs[:num_features]
    values = coefs[relevant_parameters]
    features = [node_ids[feature_names[i]].split("/")[-1] for i in relevant_parameters]

    print("Global explanation")
    for i, j in zip(features, values):
        print("{}, value: {}".format(i, j))

    sns.barplot(x="Feature", y="Importance", data=pd.DataFrame(data={"Feature": features, "Importance": values}),
                order=features).set_title("Global explanation")
    plt.show()
