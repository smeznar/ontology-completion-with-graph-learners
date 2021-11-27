import argparse

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

from models import AdamicAdar, Jaccard, Snore, node2vec, MetaPath2vec, SpectralEmbedding,\
    PreferentialAttachment, TransE, RotatE, GATModel, GINModel, GCNModel, GAEModel
from utils import read_graph_from_json, read_graph_from_txt


verbose = True
methods = {
    "Adamic": AdamicAdar,
    "Jaccard": Jaccard,
    "SNoRe": Snore,
    "node2vec": node2vec,
    "Preferential": PreferentialAttachment,
    "Spectral": SpectralEmbedding,
    "TransE": TransE,
    "RotatE": RotatE,
    "GAT": GATModel,
    "GIN": GINModel,
    "GCN": GCNModel,
    "GAE": GAEModel,
    "metapath2vec": MetaPath2vec
}


def create_splits(graph, cv=5, neg_ratio=1, val=0, seed=18):
    np.random.seed(seed)

    probs = [0 for i in range(len({e[2] for e in mgraph.edges(data="type")}))]
    edge_types = defaultdict(list)
    for i, j, k in mgraph.edges(data="type"):
        probs[k] += 1
        edge_types[(i, j)].append(k)
    probs = np.array(probs)
    etype_prob = probs / np.sum(probs)

    graph = nx.to_scipy_sparse_matrix(graph)
    graph.setdiag(0)
    graph.eliminate_zeros()
    g = graph - sp.triu(graph)
    g.eliminate_zeros()

    row, column = g.nonzero()
    perm = np.random.permutation(row.size)
    row = row[perm]
    column = column[perm]

    p_edges = [(i, j) for i, j in zip(row, column)]
    positive_edges = []
    for i, j in p_edges:
        for k in edge_types[(i, j)]+edge_types[(j, i)]:
            positive_edges.append((i, j, k))

    num_negatives = int(len(positive_edges) * neg_ratio)
    negative_edges = set()

    index_to_edge = lambda i: (i // graph.shape[1], i % graph.shape[1])
    possible_edges = graph.shape[0] * graph.shape[1]

    while len(negative_edges) < num_negatives:
        edge = np.random.randint(0, possible_edges)
        i, j = index_to_edge(edge)
        if graph[i, j] == 0 and graph[j, i] == 0 \
                and edge not in negative_edges and (j*graph.shape[1] + i) not in negative_edges and i != j:
            negative_edges.add(edge)

    negative_edges = [index_to_edge(i) for i in negative_edges]
    negative_edges = [(i, j, np.random.choice(len(etype_prob), p=etype_prob)) for i, j in negative_edges]
    cv_sets = []
    positive_in_set = int(np.ceil(len(positive_edges) / cv))
    negative_in_set = int(np.ceil(len(negative_edges) / cv))

    for i in range(cv):
        positive_test = positive_edges[(i * positive_in_set):min((i + 1) * positive_in_set, len(positive_edges))]
        negative_test = negative_edges[(i * negative_in_set):min((i + 1) * negative_in_set, len(negative_edges))]
        positive_train = positive_edges[:i * positive_in_set] + \
                         positive_edges[min((i + 1) * positive_in_set, len(positive_edges)):]
        positive_train.sort()
        negative_train = negative_edges[:i * negative_in_set] + \
                         negative_edges[min((i + 1) * negative_in_set, len(negative_edges)):]

        if val > 0:
            pos_val_nodes = int(len(positive_train) * val)
            neg_val_nodes = int(len(negative_train) * val)
            positive_val = positive_train[-pos_val_nodes:]
            positive_train = positive_train[:-pos_val_nodes]
            negative_val = negative_train[-neg_val_nodes:]
            negative_train = negative_train[:-neg_val_nodes]
        else:
            positive_val = []
            negative_val = []

        train_graph = sp.csr_matrix(([1 for _ in range(2*len(positive_train))],
                                     ([e[0] for e in positive_train] + [e[1] for e in positive_train],
                                      [e[1] for e in positive_train] + [e[0] for e in positive_train])),
                                    shape=graph.shape)
        cv_sets.append((train_graph, positive_train, negative_train,
                        positive_val, negative_val, positive_test, negative_test))

    if verbose:
        print()
        print("--------------------------------------------")
        print("Positive edges in training set: {}".format(len(positive_train)))
        print("Negative edges in training set: {}".format(len(negative_train)))
        print("Positive edges in test set: {}".format(len(positive_test)))
        print("Negative edges in test set: {}".format(len(negative_test)))
        print("Positive edges in validation set: {}".format(len(positive_val)))
        print("Negative edges in validation set: {}".format(len(negative_val)))
        print("--------------------------------------------")
        print()
    return cv_sets


def score_predictions(preds, labels):
    roc_score = roc_auc_score(labels, preds)
    ap_score = average_precision_score(labels, preds)
    return roc_score, ap_score


def write_results(roc, ap, time, name, dataset, outdir):
    if verbose:
        print()
        print("--------------------------------------------")
        print("Method: {}".format(name))
        print("ROC: {}".format(roc))
        print("AP: {}".format(ap))
        print("Time: {}".format(time))
        print("--------------------------------------------")
        print()
    with open(outdir, "w") as file:
        file.write("{}\t{}\n".format(name, dataset))
        file.write("{}\n".format("\t".join(roc)))
        file.write("{}\n".format("\t".join(ap)))
        file.write("{}\n".format("\t".join(time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link prediction benchmark')
    parser.add_argument('--method', help='Method identifier', default="SNoRe", action='store')
    parser.add_argument('--dataset', help='Name of dataset to be tested.', default="../data/anatomy.json", action='store')
    parser.add_argument('--out', help="Output directory for the score.", default="../results/_res_.txt", action='store')
    parser.add_argument('--format', help="Format of the dataset file", default="json", action='store')
    args = parser.parse_args()

    if args.format == "json":
        mgraph, node_ids, edge_ids = read_graph_from_json(args.dataset)
    elif args.format == "txt":
        mgraph, node_ids, edge_ids = read_graph_from_txt(args.dataset)
    else:
        raise Exception("Dataset format not supported")

    splits = create_splits(mgraph)

    roc_vals = []
    ap_vals = []
    time = []

    for i, split in enumerate(splits):
        if verbose:
            print("Split {}".format(i+1))
        model = methods[args.method]()
        adj, pos_train, neg_train, pos_val, neg_val, pos_test, neg_test = split
        test_edges = [(i, j) for i, j, k in pos_test] + [(i, j) for i, j, k in neg_test]
        test_labels = [1 for i in range(len(pos_test))]+[0 for i in range(len(neg_test))]

        train_mgraph = mgraph.copy()
        train_mgraph.remove_edges_from(test_edges)
        model.train(adj, pos_train, neg_train, pos_val, neg_val, train_mgraph, (len(node_ids), len(edge_ids)))
        predictions = model.predict(test_edges)

        roc, ap = score_predictions(predictions, test_labels)
        if verbose:
            print("ROC: {}, AP: {}".format(roc, ap))

        roc_vals.append(str(roc))
        ap_vals.append(str(ap))
        time.append(str(model.time))

    write_results(roc_vals, ap_vals, time, args.method, args.dataset.split("/")[-1].split(".")[0], args.out)
