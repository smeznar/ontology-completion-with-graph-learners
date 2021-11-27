import networkx as nx
import json


def read_graph_from_txt(path):
    node_ids = dict()

    graph = nx.MultiDiGraph()
    with open(path) as file:
        node_num = 0
        edge_num = 0
        nodes = dict()
        edges = dict()
        for edge in file:
            a = edge.split("\t")
            if a[0] not in nodes:
                nodes[a[0]] = node_num
                node_ids[node_num] = a[0]
                node_num += 1
            if a[1] not in nodes:
                nodes[a[1]] = node_num
                node_ids[node_num] = a[1]
                node_num += 1
            if a[2] not in edges:
                edges[a[2]] = edge_num
                edge_num += 1
            graph.add_edge(nodes[a[0]], nodes[a[1]], type=edges[a[2]])
    return graph, node_ids, edges


def read_graph_from_json(path):
    nodes = dict()
    r_n = dict()
    edge_attrs = dict()
    r_e_a = dict()
    counter = 0
    attr_counter = 0
    with open(path, "r", encoding="ISO-8859-1") as file:
        data = json.load(file)
        graph = nx.MultiDiGraph()
        network = data["graphs"]
        for n in network["nodes"]:
            if n["id"] not in nodes:
                nodes[n["id"]] = counter
                r_n[counter] = n["id"]
                graph.add_node(counter)
                counter += 1
        for e in network["edges"]:
            if e['pred'] in edge_attrs:
                num = edge_attrs[e['pred']]
            else:
                edge_attrs[e['pred']] = attr_counter
                r_e_a[attr_counter] = e["pred"]
                num = attr_counter
                attr_counter += 1
            if e["sub"] not in nodes:
                nodes[e["sub"]] = counter
                r_n[counter] = e["sub"]
                graph.add_node(counter)
                counter += 1
            if e["obj"] not in nodes:
                nodes[e["obj"]] = counter
                r_n[counter] = e["obj"]
                graph.add_node(counter)
                counter += 1
            graph.add_edge(nodes[e['sub']], nodes[e['obj']], type=num)
    return graph, r_n, r_e_a