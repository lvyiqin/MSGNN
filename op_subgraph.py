import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ops import GCN, norm_g
import os
import networkx as nx
#from numba import jit
# from numba.experimental import jitclass

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

NODE_OPS = {
  'node_add_1': lambda subs, graph: Node_func(subs, 'node_add_1', graph),
  'node_add_2': lambda subs, graph: Node_func(subs, 'node_add_2', graph),
  'node_remove_1': lambda subs, graph: Node_func(subs, 'node_remove_1', graph),
  'node_remove_2': lambda subs, graph: Node_func(subs, 'node_remove_2', graph),
  'none': lambda subs, graph: Node_func(subs, 'none', graph),
}
EDGE_OPS = {
  'edge_add_1': lambda subs, graph, x, args: Edge_func(subs, 'edge_add_1', graph, x, args),
  'edge_add_2': lambda subs, graph, x, args: Edge_func(subs, 'edge_add_2', graph, x, args),
  'edge_remove_1': lambda subs, graph, x, args: Edge_func(subs, 'edge_remove_1', graph, x, args),
  'edge_remove_2': lambda subs, graph, x, args: Edge_func(subs, 'edge_remove_2', graph, x, args),
  'none': lambda subs, graph, x, args: Edge_func(subs, 'none', graph, x, args),
}


class Node_func(nn.Module):

  def __init__(self, subs, op, graph):
    super(Node_func, self).__init__()
    self.op = op
    self.new_subs = []
    self.new_nodes = []
    self.new_subs_edges = []
    if op == 'node_add_1':
      for sub in subs:
        neighbor = []
        for j in sub:
          neighbor.extend([i for i in graph.neighbors(j)])
        nei = list(set(neighbor))
        nei = list(set(nei)-set(sub))
        if nei != []:
            new_node = np.random.choice(nei, 1).tolist()
        else:
            new_node = []
        self.new_nodes.append(new_node)
        node_sub = sub + new_node
        self.new_subs.append(node_sub)

    elif op == 'node_add_2':
      for sub in subs:
        neighbor = []
        for j in sub:
          neighbor.extend([i for i in graph.neighbors(j)])
        nei = list(set(neighbor))
        nei = list(set(nei)-set(sub))
        if nei != []:
            new_node = np.random.choice(nei, 2).tolist()
        else:
            new_node = []
        self.new_nodes.append(new_node)
        node_sub = sub + new_node
        self.new_subs.append(node_sub)

    elif op == 'node_remove_1':
      for sub in subs:
        while True:
            new_node = np.random.choice(sub, 1)
            node_sub = list(set(sub) - set(new_node.tolist()))
            if nx.is_connected(graph.subgraph(node_sub)):
                break
        self.new_nodes.append(new_node.tolist())
        self.new_subs.append(node_sub)

    elif op == 'node_remove_2':
      for sub in subs:
          if len(sub)<=2:
            self.new_subs.append(sub)
          else:
            while True:
                new_node = np.random.choice(sub, 2, replace=False)
                node_sub = list(set(sub) - set(new_node.tolist()))
                if nx.is_connected(graph.subgraph(node_sub)):
                    break
            self.new_nodes.append(new_node.tolist())
            self.new_subs.append(node_sub)

    elif op == 'none':
        self.new_subs = subs


  def forward(self, x, sub_representations):
      x = x.cuda()
      sub_representations = sub_representations.cuda()
      new_sub_representations = torch.randn(sub_representations.shape[0], sub_representations.shape[1])
      for i in range(len(self.new_nodes)):
          new_h = x[self.new_nodes[i], :]
          if self.op in ['node_add_1', 'node_add_2', 'none']:
              new_sub_representations[i] = sub_representations[i] + torch.sum(new_h, 0)
          else:
              new_sub_representations[i] = sub_representations[i] - torch.sum(new_h, 0)
      return self.new_subs, new_sub_representations



class Edge_func(nn.Module):
  def __init__(self, subs, op, graph, x, args):
    super(Edge_func, self).__init__()
    self.op = op
    self.e_new_subs = []
    self.new_nodes = []
    self.new_subs_edges = []
    self.gcn_layer = GCN(x.shape[1], args.l_dim, F.elu, args.drop_n).cuda()
    if op == 'edge_add_1':
      for sub in subs:
        neighbor = []
        H = graph.subgraph(sub)
        sub_edges = list(H.edges)
        for j in sub:
            neighbor.extend([i for i in graph.neighbors(j)])
        nei = list(set(neighbor))
        nei = list(set(nei) - set(sub))
        if nei != []:
            new_node = np.random.choice(nei, 1).tolist()
            temp = list(set(graph.neighbors(new_node[0])) & set(sub))
            new_edge = np.random.choice(temp, 1).tolist()
            node_sub = sub + new_node
            self.e_new_subs.append(list(set(node_sub)))
            sub_edges.extend([(new_node[0], ne) for ne in new_edge])
            self.new_subs_edges.append(sub_edges)
        else:
            self.new_subs_edges.append(sub_edges)
            self.e_new_subs.append(list(set(sub)))

        
    elif op == 'edge_add_2':
      for sub in subs:
        H = graph.subgraph(sub)
        assert nx.is_connected(H)
        sub_edges = list(H.edges)
        neighbor = []
        for j in sub:
            neighbor.extend([i for i in graph.neighbors(j)])
        nei_sub = list(set(sub + neighbor))
        nei_sub_edge = list(graph.subgraph(nei_sub).edges)
        rem_edge = list(set(nei_sub_edge) - set(sub_edges))
        index = list(range(len(rem_edge)))
        if sub != [] and index != []:
            new_edge_index = np.random.choice(index, 1)
            sub_edges.append(rem_edge[new_edge_index[0]])
            nsub = sub + [rem_edge[new_edge_index[0]][0]]
            nsub = nsub + [rem_edge[new_edge_index[0]][1]]
            nsub = list(set(nsub))
            
            nei_sub.extend([i for i in graph.neighbors(rem_edge[new_edge_index[0]][0])])
            nei_sub.extend([i for i in graph.neighbors(rem_edge[new_edge_index[0]][1])])
            nei_sub = list(set(nei_sub))
            nei_sub_edge = list(graph.subgraph(nei_sub).edges)
            rem_edge = list(set(nei_sub_edge) - set(sub_edges))
            index = list(range(len(rem_edge)))
            if nsub != [] and index != []:
                new_edge_index = np.random.choice(index, 1)
                sub_edges.append(rem_edge[new_edge_index[0]])
                nsub = nsub + [rem_edge[new_edge_index[0]][0]]
                nsub = nsub + [rem_edge[new_edge_index[0]][1]]
        else:
            nsub = sub
        nsub = list(set(nsub))
        assert nx.is_connected(graph.subgraph(nsub))
        self.e_new_subs.append(nsub)

        self.new_subs_edges.append(sub_edges)  # [[(5, 4), (1, 3), (3, 4), (9, 4), (1, 2), (1, 0)], [(8, 9), (10, 8), (7, 8), (6, 7), (7, 12), (11, 10)],...]

    elif op == 'edge_remove_1':
      for sub in subs:
        H = graph.subgraph(sub)
        sub_edges = list(H.edges)
        index = list(range(len(sub_edges)))
        if len(index) <= 1:
            self.e_new_subs.append(sub)
            self.new_subs_edges.append(sub_edges)
        else:
            while True:
                new_edge_index = np.random.choice(index, 1)
                sub_edges_remove = list(set(sub_edges) - set([sub_edges[new_edge_index[0]]]))
                h = graph.edge_subgraph(sub_edges_remove)
                if nx.is_connected(h):
                    break
            sub_node = list(h.nodes())
            self.e_new_subs.append(sub_node)
            self.new_subs_edges.append(sub_edges_remove)

    elif op == 'edge_remove_2':
      for sub in subs:
        H = graph.subgraph(sub)
        sub_edges = list(H.edges)
        index = list(range(len(sub_edges)))
        if len(index) <= 2:
            self.e_new_subs.append(sub)
            self.new_subs_edges.append(sub_edges)
        else:
            while True:
                new_edge_index = np.random.choice(index, 2, replace=False)
                a = sub_edges[new_edge_index[0]]
                b = sub_edges[new_edge_index[1]]
                sub_edges_remove = list(set(sub_edges) - set([a, b]))
                h = graph.edge_subgraph(sub_edges_remove)
                if nx.is_connected(h):
                    break
            sub_node = list(h.nodes())
            self.e_new_subs.append(sub_node)
            self.new_subs_edges.append(sub_edges_remove)

    elif op == 'none':
      for sub in subs:
        H = graph.subgraph(sub)
        sub_edges = list(H.edges)
        self.new_subs_edges.append(sub_edges)
      self.e_new_subs = subs

  def forward(self, x, graph, args):
      x = x.cuda()
      new_sub_representations = torch.randn(len(self.e_new_subs), args.l_dim)
      assert len(self.e_new_subs) == len(self.new_subs_edges)
      for i in range(len(self.e_new_subs)):
        sub = self.e_new_subs[i]
        sub = list(set(sub))
        edge = self.new_subs_edges[i]
        if sub != [] and edge != []:
            new_h = x[sub, :]
            H = graph.edge_subgraph(edge)
            new_g = nx.to_numpy_matrix(H)
            new_g = torch.from_numpy(new_g).to(torch.float32).cuda()
  
            new_h = self.gcn_layer(norm_g(new_g), new_h)
            layer_norm = nn.LayerNorm(normalized_shape=new_h.size(),elementwise_affine=False)
            new_h = layer_norm(new_h)
            new_h = F.dropout(new_h, p=args.drop_n)
            sub_representations = torch.sum(new_h, 0)
            new_sub_representations[i] = sub_representations
      return self.e_new_subs, new_sub_representations



