import random
import copy
import networkx as nx
from functools import reduce
import operator
import dgl


def find_neighbor(j, graph, eps):
    N = list()
    temp = graph.neighbors(j)
    if len(temp) >= eps:
        N.append(temp)
    return set(N)


def DBSCAN(graph, min_Pts):
    graph = graph.to_networkx()
    k = -1
    neighbor_list = []  
    omega_list = []  
    X = len(graph.nodes())
    gama = set([x for x in range(X)])  
    clusters = [-1 for _ in range(X)] 

    for i in range(X):
        neighbor_list.append(set(graph.neighbors(i)))
        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i)  
    omega_list = set(omega_list)  
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q]) >= min_Pts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            clusters[Cklist[i]] = k
        omega_list = omega_list - Ck
    return clusters


def cluster(sub_set, graph):
    s = []
    e = []
    for edge in graph.edges():
        s.append(edge[0])
        e.append(edge[1])
    graph.remove_edges_from(nx.selfloop_edges(graph))
    pool_graph = list(set(reduce(operator.add, sub_set)))
    remain_nodes = list(set(list(graph.nodes())) - set(pool_graph))
    dgl_graph = dgl.from_networkx(graph)
    sg = dgl.node_subgraph(dgl_graph, remain_nodes)
    clu = DBSCAN(sg, min_Pts=2)
    return clu, remain_nodes

