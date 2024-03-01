import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import sys

sys.setrecursionlimit(10**6)

SUB_size = 10


class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i]))
            
    def forward(self, g, h):
        indices_list = []
        # pool
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            g, h, idx, subs1 = self.pools[i](g, h)
            indices_list.append(idx)

        h = self.bottom_gcn(g, h)
        return h


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h


class Pool(nn.Module):
    def __init__(self, k):
        super(Pool, self).__init__()
        self.k = k

    def forward(self, g, h):
        sub = Subgraph(self.k)
        return sub.subgraphs(g, h)


class Subgraph(nn.Module):

    def __init__(self):
        super(Subgraph, self).__init__()
        
    def huge_star(self, graph):
        t = nx.triangles(graph)
        temp = {k: v for k, v in t.items() if v == 0}
        star = (list(temp.keys()))
        star_deg = {i: graph.degree(i) for i in star}
        te = dict(sorted(star_deg.items(), key=lambda x: x[1], reverse=True))
        h_star = list(te)[:int(0.5 * len(temp))]
        return h_star

    def super_pivot(self, graph):
        t = nx.triangles(graph)
        temp = {k: v for k, v in t.items() if v != 0}
        pivot = list(temp.keys())

        pivot_deg = {i: graph.degree(i) for i in pivot}
        te = dict(sorted(pivot_deg.items(), key=lambda x: x[1], reverse=True))
        s_pivot = list(te)[:int(0.05 * len(temp))]
        return s_pivot


    def cuttingpoint(self, graph):
        edges = graph.edges()
        edges = [(a, b) for a, b in edges]
        link, dfn, low = {}, {}, {}
        global_time = [0]
        for (a, b) in edges:
            if a not in link:
                link[a] = []
            if b not in link:
                link[b] = []
            link[a].append(b) 
            link[b].append(a) 
            dfn[a], dfn[b] = 0x7fffffff, 0x7fffffff
            low[a], low[b] = 0x7fffffff, 0x7fffffff

        cutting_points, cutting_edges = [], []

        def dfs(cur, prev, root):
            global_time[0] += 1
            dfn[cur], low[cur] = global_time[0], global_time[0]

            children_cnt = 0
            flag = False
            for next in link[cur]:
                if next != prev:
                    if dfn[next] == 0x7fffffff:
                        children_cnt += 1
                        dfs(next, cur, root)

                        if cur != root and low[next] >= dfn[cur]:
                            flag = True
                        low[cur] = min(low[cur], low[next])

                        if low[next] > dfn[cur]:
                            cutting_edges.append([cur, next] if cur < next else [next, cur])
                    else:
                        low[cur] = min(low[cur], dfn[next])

            if flag or (cur == root and children_cnt >= 2):
                cutting_points.append(cur)
        dfs(edges[0][0], None, edges[0][0])
        return cutting_points


    def bfs(self, graph, s):
        queue = []
        queue.append(s)  
        seen = set()  
        seen.add(s) 
        while (len(queue) > 0):
            vertex = queue.pop(0)  
            nodes = graph.neighbors(vertex) 
            for w in nodes:
                if w not in seen:
                    queue.append(w) 
                    seen.add(w)  
                    if len(seen) == SUB_size:
                        return list(seen)
        return list(seen)


    def subgraphs(self, g, h):
        subs = []
        n_g = g.cpu().numpy()
        graph = nx.from_numpy_array(n_g)
        
        h_star = self.huge_star(graph)
        s_pivot = self.super_pivot(graph)
        cutting_points = self.cuttingpoint(graph)
        
        idx = list(set(h_star + s_pivot + cutting_points))
        for s in idx:
            subs.append(self.bfs(graph, s))
        assert (graph.number_of_nodes() == g.shape[0])
        new_h = h[idx, :]
        un_g = g.bool().float()

        un_g = torch.matmul(un_g, un_g).bool().float()
        un_g = un_g[idx, :]
        un_g = un_g[:, idx]
        g = norm_g(un_g)
        return g, new_h, idx, subs


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
