from abc import ABC, abstractmethod
import time

import numpy as np
import networkx as nx
import torch_geometric.utils
from sklearn.manifold import spectral_embedding
from gensim.models import Word2Vec
import torch
import tqdm
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric as tg
from snore import SNoRe
from stellargraph import StellarDiGraph
from stellargraph.data import UniformRandomMetaPathWalk

import node2vec as n2v


class PredictionModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        pass

    @abstractmethod
    def predict(self, test_edges):
        pass


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


class Jaccard(PredictionModel):
    def __init__(self):
        self.time = 0
        self.nx_graph = None

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        self.nx_graph = nx.from_scipy_sparse_matrix(adj)
        self.nx_graph.to_undirected()

    def predict(self, test_edges):
        start_time = time.time()
        adar = nx.jaccard_coefficient(self.nx_graph, ebunch=test_edges)
        pred = np.array([i[2] for i in adar])

        self.time = time.time() - start_time
        return pred / np.max(pred)


class PreferentialAttachment(PredictionModel):
    def __init__(self):
        self.time = 0
        self.nx_graph = None

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        self.nx_graph = nx.from_scipy_sparse_matrix(adj)
        self.nx_graph.to_undirected()

    def predict(self, test_edges):
        start_time = time.time()
        adar = nx.preferential_attachment(self.nx_graph, ebunch=test_edges)
        pred = np.array([i[2] for i in adar])

        self.time = time.time() - start_time
        return pred / np.max(pred)


class Snore(PredictionModel):
    def __init__(self):
        self.model = SNoRe()
        self.embedding = None
        self.time = 0

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        start_time = time.time()
        self.embedding = self.model.embed(adj)
        self.time = time.time() - start_time

    def predict(self, test_edges):
        pred = []
        for n1, n2 in test_edges:
            pred.append(self.embedding[n1].toarray().dot(self.embedding[n2].toarray().T))
        pred = np.array(pred).squeeze()

        return pred / np.max(pred)


class node2vec(PredictionModel):
    def __init__(self):
        self.p = 1
        self.q = 1
        self.window_size = 10
        self.num_walks = 10
        self.walk_length = 80
        self.dim = 128
        self.workers = 1
        self.iters = 1
        self.seed = 18
        self.time = 0
        self.embedding = None

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        start_time = time.time()
        np.random.seed(self.seed)
        g_n2v = n2v.Graph(nx.from_scipy_sparse_matrix(adj), False, self.p, self.q, self.seed)
        g_n2v.preprocess_transition_probs()
        walks = g_n2v.simulate_walks(self.num_walks, self.walk_length, verbose=False)
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, vector_size=self.dim, window=self.window_size, min_count=0,
                         sg=1, workers=self.workers, epochs=self.iters, seed=self.seed)
        emb_mappings = model.wv

        emb_list = []
        for node_index in range(0, adj.shape[0]):
            node_str = str(node_index)
            node_emb = emb_mappings[node_str]
            emb_list.append(node_emb)
        self.embedding = np.vstack(emb_list)
        self.time = time.time() - start_time

    def predict(self, test_edges):
        preds = []
        for n1, n2 in test_edges:
            preds.append(self.embedding[n1].dot(self.embedding[n2].T))
        preds = np.array(preds).squeeze()

        return preds / np.max(preds)


class TransEModel(torch.nn.Module):
    def __init__(self, num1, num2, output_dim, gamma):
        super(TransEModel, self).__init__()
        self.emb_ent_real = torch.nn.Embedding(num1, output_dim)  # real
        # Real embeddings of relations.
        self.emb_rel_real = torch.nn.Embedding(num2, output_dim)
        self.gamma = torch.nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.init()
        self.loss_f = torch.nn.BCELoss()

    def forward(self, x, rel=None):
        emb_head = self.emb_ent_real(x[:, 0])
        if rel is None:
            emb_rel = self.emb_rel_real(x[:, 2])
        else:
            emb_rel = self.emb_rel_real(torch.tensor([rel]*x.size(0)).long())
        emb_tail = self.emb_ent_real(x[:, 1])
        distance = torch.norm((emb_head + emb_rel) - emb_tail, p=1, dim=1)
        score = self.gamma.item() - distance
        return torch.sigmoid(score)

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_ent_real.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel_real.weight.data)

    def loss(self, score, target):
        return self.loss_f(score, target)


class TransE(PredictionModel):
    """
    TransE trained with binary cross entropy
    """

    def __init__(self):
        super(TransE, self).__init__()
        self.time = 0
        self.dim = 128
        self.epochs = 180
        self.batch_size = 128
        self.seed = 18
        torch.manual_seed(self.seed)

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        x = torch.cat([torch.tensor(pos_edges), torch.tensor(neg_edges)])
        y = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))]).float()
        trainset = torch.utils.data.TensorDataset(x, y)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        st = time.time()
        self.num_rel = classes[1]
        self.model = TransEModel(classes[0], classes[1], self.dim, 0.0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=9 * 10 ** -4)
        for epoch in range(self.epochs):
            total_loss = 0
            counter = 0
            with tqdm.tqdm(total=len(trainloader.dataset), desc="Epoch: {}".format(epoch), unit='chunks') as prog_bar:
                for i in range(len(trainloader)):
                    data = next(iter(trainloader))
                    x = self.model(data[0])
                    counter += 1
                    loss = torch.clamp(self.model.loss(x, data[1]), min=0., max=50000.).double()
                    total_loss += loss
                    prog_bar.set_postfix(**{'run:': "TransE",
                                            'loss': loss.item()})
                    prog_bar.update(self.batch_size)
            optimizer.zero_grad()
            total_loss /= counter
            # print(total_loss.item())
            total_loss.backward()
            optimizer.step()
        self.time = time.time() - st

    def predict(self, test_edges):
        x_test = torch.tensor(test_edges)
        scores = [self.model(x_test, rel=i) for i in range(self.num_rel)]
        preds = torch.max(torch.stack(scores), dim=0).values.detach().numpy()

        return preds


class RotatEModel(torch.nn.Module):
    def __init__(self, num1, num2, output_dim, gamma):
        super(RotatEModel, self).__init__()
        self.emb_ent_real = torch.nn.Embedding(num1, output_dim)
        self.emb_ent_img = torch.nn.Embedding(num1, output_dim)
        self.emb_rel = torch.nn.Embedding(num2, output_dim)
        self.gamma = torch.nn.Parameter(
            torch.tensor([gamma]),
            requires_grad=False
        )
        self.embedding_range = torch.nn.Parameter(
            torch.tensor([(gamma + 2.0) / output_dim]),
            requires_grad=False
        )
        self.phase = torch.nn.Parameter(self.embedding_range / torch.tensor(np.pi).float(), requires_grad=False)
        self.init()
        self.loss_f = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_ent_real.weight.data)
        torch.nn.init.xavier_normal_(self.emb_ent_img.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, x, rel=None):
        head_real = self.emb_ent_real(x[:, 0])
        tail_real = self.emb_ent_real(x[:, 1])
        head_img = self.emb_ent_img(x[:, 0])
        tail_img = self.emb_ent_img(x[:, 1])
        if rel is None:
            emb_rel = self.emb_rel(x[:, 2])
        else:
            emb_rel = self.emb_rel(torch.tensor([rel]*x.size(0)).long())

        phase_relation = emb_rel / self.phase
        rel_real = torch.cos(phase_relation)
        rel_img = torch.sin(phase_relation)

        real_score = (head_real * rel_real - head_img * rel_img) - tail_real
        img_score = (head_real * rel_img + head_img * rel_real) - tail_img

        score = torch.stack([real_score, img_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=1)
        return torch.sigmoid(score)

    def loss(self, score, target):
        return self.loss_f(score, target)


class RotatE(PredictionModel):
    """
    TransE trained with binary cross entropy
    """

    def __init__(self):
        super(RotatE, self).__init__()
        self.time = 0
        self.dim = 128
        self.epochs = 180
        self.batch_size = 128
        self.seed = 18
        torch.manual_seed(self.seed)

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        x = torch.cat([torch.tensor(pos_edges), torch.tensor(neg_edges)])
        y = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))]).float()
        trainset = torch.utils.data.TensorDataset(x, y)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        st = time.time()
        self.num_rel = classes[1]
        self.model = RotatEModel(classes[0], classes[1], self.dim, 0.0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=9 * 10 ** -4)
        for epoch in range(self.epochs):
            total_loss = 0
            counter = 0
            with tqdm.tqdm(total=len(trainloader.dataset), desc="Epoch: {}".format(epoch), unit='chunks') as prog_bar:
                for i in range(len(trainloader)):
                    data = next(iter(trainloader))
                    x = self.model(data[0])
                    counter += 1
                    loss = torch.clamp(self.model.loss(x, data[1]), min=0., max=50000.).double()
                    total_loss += loss
                    prog_bar.set_postfix(**{'run:': "RotatE",
                                            'loss': loss.item()})
                    prog_bar.update(self.batch_size)
            optimizer.zero_grad()
            total_loss /= counter
            # print(total_loss.item())
            total_loss.backward()
            optimizer.step()
        self.time = time.time() - st

    def predict(self, test_edges):
        x_test = torch.tensor(test_edges)
        scores = [self.model(x_test, rel=i) for i in range(self.num_rel)]
        preds = torch.max(torch.stack(scores), dim=0).values.detach().numpy()
        return preds


class MetaPath2vec(PredictionModel):
    def __init__(self):
        self.p = 1
        self.q = 1
        self.window_size = 10
        self.num_walks = 10
        self.walk_length = 80
        self.dim = 128
        self.workers = 1
        self.iters = 1
        self.time = 0
        self.seed = 18
        self.embedding = None

    def train(self, adj, pos_edges, neg_edges, pos_test, val_neg, mgraph, classes):
        start_time = time.time()
        mgraph.remove_edges_from(pos_test)
        sg = StellarDiGraph.from_networkx(mgraph, edge_type_attr="type")
        metapaths = [["default"]*80]
        rw = UniformRandomMetaPathWalk(sg, seed=self.seed)
        walks = rw.run(
            nodes=list(sg.nodes()),  # root nodes
            length=self.walk_length,  # maximum length of a random walk
            n=self.num_walks,  # number of random walks per root node
            metapaths=metapaths,  # the metapaths
        )

        model = Word2Vec(walks, vector_size=self.dim, window=self.window_size, min_count=0,
                         sg=1, workers=self.workers, epochs=self.iters, seed=self.seed)
        emb_mappings = model.wv

        emb_list = []
        for node_index in range(0, len(mgraph.nodes)):
            node_str = node_index
            node_emb = emb_mappings[node_str]
            emb_list.append(node_emb)
        self.embedding = np.vstack(emb_list)
        self.time = time.time() - start_time

    def predict(self, test_edges):
        preds = []
        for n1, n2 in test_edges:
            preds.append(self.embedding[n1].dot(self.embedding[n2].T))
        preds = np.array(preds).squeeze()

        return preds / np.max(preds)


class SpectralEmbedding(PredictionModel):
    def __init__(self):
        self.embedding = None
        self.random_state = 18
        self.time = 0

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        start_time = time.time()
        self.embedding = spectral_embedding(adj, n_components=16, random_state=self.random_state)
        self.time = time.time() - start_time

    def predict(self, test_edges):
        preds = []
        for n1, n2 in test_edges:
            preds.append(self.embedding[n1].dot(self.embedding[n2].T))
        preds = np.array(preds).squeeze()

        return preds / np.max(preds)


def get_nn_data(pos_edges, neg_edges):
    pos = torch.tensor([[e[0] for e in pos_edges], [e[1] for e in pos_edges]])
    neg = torch.tensor([[e[0] for e in neg_edges], [e[1] for e in neg_edges]])
    all_edges = torch.cat([pos, neg], dim=1)
    pos_lab = torch.tensor([1 for i in range(len(pos_edges))], dtype=torch.float)
    neg_lab = torch.tensor([0 for i in range(len(neg_edges))], dtype=torch.float)
    all_labels = torch.cat([pos_lab, neg_lab])
    # permutation = np.random.permutation((len(pos_edges)+len(neg_edges)))
    # all_edges = all_edges[:, torch.from_numpy(permutation)]
    # all_labels = all_labels[torch.from_numpy(permutation)]
    return pos, all_edges, all_labels


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, size=None, dropout=0.1):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = tg.nn.GATConv(input_dim, hidden_dim, add_self_loops=False)
        self.conv2 = tg.nn.GATConv(hidden_dim, output_dim, add_self_loops=False)
        self.size = size

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GATModel(PredictionModel):
    def __init__(self):
        self.model = None
        self.epochs = 300
        self.hidden_dim = 128
        self.output_dim = 128
        self.seed = 18
        torch.manual_seed(self.seed)

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        self.adj = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
        self.num_nodes = adj.shape[0]
        train_pos, all_edges, all_labels = get_nn_data(pos_edges, neg_edges)

        self.model = GAT(self.num_nodes, self.hidden_dim, self.output_dim, size=adj.shape)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        loss_func = torch.nn.BCEWithLogitsLoss()
        start_time = time.time()
        self.x = torch.eye(self.num_nodes)

        for i in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            avg_loss = 0
            out = self.model(self.x, self.adj)
            nodes_first = torch.index_select(out, 0, all_edges[0, :])
            nodes_second = torch.index_select(out, 0, all_edges[1, :])
            pred = torch.sum(nodes_first * nodes_second, dim=-1)
            loss = loss_func(pred, all_labels)
            loss.backward()
            avg_loss += loss
            print("Epoch {}, loss {}".format(i, loss))
            optimizer.step()
            optimizer.zero_grad()

        self.time = time.time() - start_time

    def predict(self, test_edges):
        with torch.no_grad():
            pos = torch.tensor([[e[0] for e in test_edges], [e[1] for e in test_edges]], dtype=torch.long)
            out = self.model(self.x, self.adj)
            nodes_first = torch.index_select(out, 0, pos[0, :])
            nodes_second = torch.index_select(out, 0, pos[1, :])
            pos_pred = torch.sum(nodes_first * nodes_second, dim=-1).numpy()
        return pos_pred


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.conv1_input = Linear(input_dim, hidden_dim)
        self.conv1 = tg.nn.GINConv(self.conv1_input)
        self.conv2_input = Linear(hidden_dim, output_dim)
        self.conv2 = tg.nn.GINConv(self.conv2_input)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GINModel(PredictionModel):
    def __init__(self):
        self.model = None
        self.epochs = 300
        self.hidden_dim = 128
        self.output_dim = 128
        self.seed = 18
        torch.manual_seed(self.seed)

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        self.adj = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
        self.num_nodes = adj.shape[0]
        train_pos, all_edges, all_labels = get_nn_data(pos_edges, neg_edges)

        self.model = GIN(self.num_nodes, self.hidden_dim, self.output_dim)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        loss_func = torch.nn.BCEWithLogitsLoss()
        start_time = time.time()
        self.x = torch.eye(self.num_nodes)

        for i in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            avg_loss = 0
            out = self.model(self.x, self.adj)
            nodes_first = torch.index_select(out, 0, all_edges[0, :])
            nodes_second = torch.index_select(out, 0, all_edges[1, :])
            pred = torch.sum(nodes_first * nodes_second, dim=-1)
            loss = loss_func(pred, all_labels)
            loss.backward()
            avg_loss += loss
            print("Epoch {}, loss {}".format(i, loss))
            optimizer.step()
            optimizer.zero_grad()
        self.time = time.time() - start_time

    def predict(self, test_edges):
        with torch.no_grad():
            pos = torch.tensor([[e[0] for e in test_edges], [e[1] for e in test_edges]], dtype=torch.long)
            out = self.model(self.x, self.adj)
            nodes_first = torch.index_select(out, 0, pos[0, :])
            nodes_second = torch.index_select(out, 0, pos[1, :])
            pos_pred = torch.sum(nodes_first * nodes_second, dim=-1).numpy()
        return pos_pred


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.conv1 = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GCNModel(PredictionModel):
    def __init__(self):
        self.model = None
        self.epochs = 300
        self.hidden_dim = 128
        self.output_dim = 128
        self.seed = 18
        torch.manual_seed(self.seed)

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        self.adj = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
        self.num_nodes = adj.shape[0]
        train_pos, all_edges, all_labels = get_nn_data(pos_edges, neg_edges)

        self.model = GCN(self.num_nodes, self.hidden_dim, self.output_dim)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        loss_func = torch.nn.BCEWithLogitsLoss()
        start_time = time.time()
        self.x = torch.eye(self.num_nodes)

        for i in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            avg_loss = 0
            out = self.model(self.x, self.adj)
            nodes_first = torch.index_select(out, 0, all_edges[0, :])
            nodes_second = torch.index_select(out, 0, all_edges[1, :])
            pred = torch.sum(nodes_first * nodes_second, dim=-1)
            loss = loss_func(pred, all_labels)
            loss.backward()
            avg_loss += loss
            print("Epoch {}, loss {}".format(i, loss))
            optimizer.step()
            optimizer.zero_grad()
        self.time = time.time() - start_time

    def predict(self, test_edges):
        with torch.no_grad():
            pos = torch.tensor([[e[0] for e in test_edges], [e[1] for e in test_edges]], dtype=torch.long)
            out = self.model(self.x, self.adj)
            nodes_first = torch.index_select(out, 0, pos[0, :])
            nodes_second = torch.index_select(out, 0, pos[1, :])
            pos_pred = torch.sum(nodes_first * nodes_second, dim=-1).numpy()
        return pos_pred


class GCNGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNGAE, self).__init__()
        self.conv1 = tg.nn.GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = tg.nn.GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GAEModel(PredictionModel):
    def __init__(self):
        self.model = None
        self.epochs = 300
        self.hidden_dim = 128
        self.output_dim = 128
        self.seed = 18
        torch.manual_seed(self.seed)

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        self.adj = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
        self.num_nodes = adj.shape[0]
        train_pos, all_edges, all_labels = get_nn_data(pos_edges, neg_edges)

        self.model = tg.nn.GAE(GCNGAE(self.num_nodes, self.hidden_dim, self.output_dim))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        loss_func = torch.nn.BCEWithLogitsLoss()
        start_time = time.time()
        self.x = torch.eye(self.num_nodes)

        for i in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            avg_loss = 0
            out = self.model.encode(self.x, self.adj)
            nodes_first = torch.index_select(out, 0, all_edges[0, :])
            nodes_second = torch.index_select(out, 0, all_edges[1, :])
            pred = torch.sum(nodes_first * nodes_second, dim=-1)
            loss = loss_func(pred, all_labels)
            loss.backward()
            avg_loss += loss
            print("Epoch {}, loss {}".format(i, loss))
            optimizer.step()
            optimizer.zero_grad()
        self.time = time.time() - start_time

    def predict(self, test_edges):
        with torch.no_grad():
            pos = torch.tensor([[e[0] for e in test_edges], [e[1] for e in test_edges]], dtype=torch.long)
            out = self.model.encode(self.x, self.adj)
            nodes_first = torch.index_select(out, 0, pos[0, :])
            nodes_second = torch.index_select(out, 0, pos[1, :])
            pos_pred = torch.sum(nodes_first * nodes_second, dim=-1).numpy()
        return pos_pred