import argparse
import os
import configparser
import re
import torch
import networkx as nx
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical


######################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MUTAG", dest='dataset')
args = parser.parse_known_args()[0]
######################################################################

os.chdir('dataset/{}'.format(args.dataset))
config = configparser.ConfigParser()
config.read('.config')
getConfig = lambda x: config.get('config', x)
ds = getConfig('ds')
cate = int(getConfig('cate'))
class_size = int(getConfig('class_size'))


class FileLoader(object):
    def __init__(self, args):
        self.args = args
        self.ds = ds
        self.cate = cate
        self.class_size = class_size

        with open(self.ds + '_A.txt', 'r') as f:
            a = f.readlines()
            self.m = len(a)

        with open(self.ds + '_graph_labels.txt', 'r') as f:
            a = f.readlines()
            self.N = len(a)

        with open(self.ds + '_node_labels.txt', 'r') as f:
            a = f.readlines()
            self.nodeIndex2label = {}
            for index, item in enumerate(a):
                self.nodeIndex2label[index] = int(item)
            self.n = len(a)

        print("m:{},N:{},n:{}".format(self.m, self.N, self.n))

    def gen_graph(self, d_gns, d_es, d_gl, labs):
        g_list = []
        a = 0
        for i in d_gns:
            # create  graph
            g = nx.Graph()
            edges = []
            adj_n = len(d_gns[i])
            adj = torch.zeros((adj_n, adj_n))
            temp = self.dict_slice(d_es, a, a+len(d_gns[i]))

            sd_es = {k: [] for k in range(len(d_gns[i]))}

            for j in range(len(temp)):
                for jj in temp[j+a]:
                    sd_es[j].append(jj - a)
            # print(sd_es[0])
            a += len(d_gns[i])
            for it in sd_es:
                for its in sd_es[it]:
                    adj[it][its] = 1
                    edges.append(tuple(sorted([it, its])))
            edges = list(set(edges))
            g.add_edges_from(edges)
            # Remove orphaned nodes
            g.remove_nodes_from(list(nx.isolates(g)))
            #assert(g.number_of_nodes() == adj_n)

            # process graph
            g.label = d_gl[i]
            A = adj + torch.eye(adj_n)
            g.A = A
            feas = torch.tensor(labs[i])
            g.feas = feas
            g_list.append(g)

        return g_list

    def dict_slice(self, d_es, start, end):
        keys = d_es.keys()
        dict_slice = {}
        for k in list(keys)[start:end]:
            dict_slice[k] = d_es[k]
        return dict_slice

    # Return a dictionary, key: the number of the graph, value: the number of the nodes contained in the graph, which is a list (all starting from 0)
    def ex_which_graph(self):
        d_n2g = {}
        with open(self.ds + '_graph_indicator.txt') as f:
            index = 0
            line = f.readline()
            while line:
                d_n2g[index] = int(line)
                index += 1
                line = f.readline()
        d_gns = {k: [] for k in range(self.N)}
        for it in d_n2g:
            d_gns[d_n2g[it] - 1].append(it)
        return d_gns

    # Return a dictionary, key: node number, starting from 0, value: list, element is the number of nodes with an edge between the node
    def ex_edges(self):
        d_es = {k: [] for k in range(self.n)}
        with open(self.ds + '_A.txt') as f:
            line = f.readline()
            while line:
                line = line.replace(' ', '')
                s = re.search('[0-9]{1,}', line).group()
                d = re.search('[,][0-9]{1,}', line).group()
                d = d.strip(',')
                d_es[int(s) - 1].append(int(d) - 1)
                line = f.readline()
        return d_es

    def ex_g_labels(self):
        d_gl = []
        with open(self.ds + '_graph_labels.txt') as f:
            line = f.readline()
            while line:
                if int(line) == -1:
                    d_gl.append(int(line) + 1)
                else:
                    d_gl.append(int(line))

                line = f.readline()
        return d_gl

    def ex_n_labels(self, d_gns):
        labs = {k: [] for k in range(self.N)}

        with open(self.ds + '_node_labels.txt') as f:
            node_tags = []
            feat_dict = {}
            line = f.readline()
            while line:
                if int(line) not in feat_dict:
                    feat_dict[int(line)] = len(feat_dict)
                node_tags.append(feat_dict[int(line)])
                line = f.readline()

        i = 0
        for graph in d_gns:
            labs[i] = to_categorical(
                [node_tags[it] for it in d_gns[graph]], self.cate)
            i += 1
        return labs

    def load_data(self):
        args = self.args
        print('loading data ...')
        d_gns = self.ex_which_graph()

        d_es = self.ex_edges()
        d_gl = self.ex_g_labels()
        labs = self.ex_n_labels(d_gns)
        g_list = self.gen_graph(d_gns, d_es, d_gl, labs)
        print('g_list:',len(g_list))
        return G_data(self.class_size, self.cate, g_list), g_list


class SubFileloader(object):
    def __init__(self, gall, hall, labels, node_subs):
        self.gall = gall
        self.hall = hall
        self.node_subs = node_subs
        self.labels = labels
        # print('hall_len:', len(hall))
        # print('gall_len:', len(gall))
        # print('nodesublen:,',len(node_subs))

    def load_subdate(self):
        g_list = []
        for i in range(len(self.gall)):
            g = nx.Graph()
            g.adjs = self.gall[i]
            g.embs = self.hall[i]
            g.subs = [] if self.node_subs == [] else self.node_subs[i]
            g.graphs = nx.from_numpy_array(np.array(self.gall[i].cpu()))
            g.label = self.labels[i]
            g_list.append(g)
        return G_data(class_size, cate, g_list), g_list


class G_data(object):
    def __init__(self, num_class, feat_dim, g_list):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.g_list = g_list
        self.sep_data()

    def sep_data(self, seed=0):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        labels = [g.label for g in self.g_list]
        # print('labels:', labels)
        self.idx_list = list(skf.split(np.zeros(len(labels)), labels))

    def use_fold_data(self, fold_idx):
        self.fold_idx = fold_idx+1
        train_idx, test_idx = self.idx_list[fold_idx]
        self.train_gs = [self.g_list[i] for i in train_idx]
        self.test_gs = [self.g_list[i] for i in test_idx]
