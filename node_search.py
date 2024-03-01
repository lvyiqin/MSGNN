import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from genotypes import NODE_PRIMITIVES, EDGE_PRIMITIVES, COMBINE_PRIMITIVES
from op_subgraph import *
from ops import GCN, Subgraph
import argparse
from model_search import Top_k_Subs
from discriminator import Discriminator

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-data', default='NCI109', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=100, help='epochs')
    parser.add_argument('-batch', type=int, default=32, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=2, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=128, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.3, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', default=[0.6, 0.5])
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5.979931414632729e-05, help='weight decay')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--MIepochs', type=int, default=5, help='num of training epochs')
    parser.add_argument('--temp', type=float, default=0.2, help=' temperature in gumble softmax.')
    parser.add_argument('--loc_mean', type=float, default=10.0, help='initial mean value to generate the location')
    parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
    parser.add_argument('--arch_learning_rate', type=float, default=0.08, help='learning rate for arch encoding')
    parser.add_argument('--arch_learning_rate_min', type=float, default=0.005,
                        help='minimum learning rate for arch encoding')
    parser.add_argument('--learning_rate', type=float, default=0.007662125400401121, help='init learning rate')
    parser.add_argument('--model_type', type=str, default='snas', help='how to update alpha', choices=['darts', 'snas'])
    parser.add_argument('--w_update_epoch', type=int, default=1, help='epoches in update W')
    args, _ = parser.parse_known_args()
    return args


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CNN(nn.Module):
    def __init__(self, sub_len, in_dim):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=sub_len-2+1))

    def forward(self, x):
        if min(x.shape) <= 1:
            if min(x.shape) == 0:
                return torch.zeros(1,x.shape[-1]).cuda()
            else:
                x = torch.squeeze(x,0)
                return torch.max(x,dim=0)[0]
        else:
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return x


class NodeMixedOp(nn.Module):
    def __init__(self, subs, graph, weights):
        super(NodeMixedOp, self).__init__()
        self.weights = weights
        self._ops = nn.ModuleList()
        for primitive in NODE_PRIMITIVES:
            op = NODE_OPS[primitive](subs, graph)
            self._ops.append(op)

    def forward(self, x, sub_representations):
        mixed_res = []
        for w, op in zip(self.weights[0], self._ops):
            tensor = op(x, sub_representations)[1]
            tensor = tensor.cuda()
            assert tensor.shape == sub_representations.shape
            cnn = CNN(sub_representations.shape[0], sub_representations.shape[1])
            cnn = cnn.cuda()
            mixed_res.append(w * cnn(tensor.unsqueeze(0)))
        if len(sum(mixed_res)) != 1:
            mixed = sum(mixed_res).unsqueeze(0)
        else:
            mixed = sum(mixed_res)
        return mixed


class EdgeMixedOp(nn.Module):
    def __init__(self, subs, graph, weights, x, args):
        super(EdgeMixedOp, self).__init__()
        self.weights = weights
        
        self._ops = nn.ModuleList()
        self.cnn_layer = CNN(len(subs), args.l_dim).cuda()
        
        for primitive in EDGE_PRIMITIVES:
            op = EDGE_OPS[primitive](subs, graph, x, args)
            self._ops.append(op)
    def forward(self, x, g, args):
        mixed_res = []
        for w, op in zip(self.weights[0], self._ops):
            tensor = op(x, g, args)[1]
            tensor = tensor.cuda()
            mixed_res.append(w * self.cnn_layer(tensor.unsqueeze(0)))
        if len(sum(mixed_res)) != 1:
            mixed = sum(mixed_res).unsqueeze(0)
        else:
            mixed = sum(mixed_res)
        return mixed


class NodeSearch(nn.Module):
    def __init__(self, k, in_dim, hidden_size, args):
        super(NodeSearch, self).__init__()
        self.args = args
        self.gcn_layer = GCN(in_dim, self.args.l_dim, F.elu, self.args.drop_n)
        # subgraphs select
        self.select_layers = Top_k_Subs(k, self.args.l_dim, self.args.drop_n)

        self.disc = Discriminator(self.args.l_dim)
        self._initialize_alphas()


        # node search
    def _get_categ_mask(self, alpha):
        log_alpha = alpha
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self.args.temp)
        return one_hot


    def forward(self, original, x, g, graph):
        if self.args.model_type == 'darts':
            node_alphas = F.softmax(self.log_node_alphas, dim=-1)
        else:
            node_alphas = self._get_categ_mask(self.log_node_alphas)
            
        new_subs_n = []
        bs = []
        hbatch = []
        node_indices = torch.argmax(node_alphas, dim=-1)
        gene_node = NODE_PRIMITIVES[node_indices]
        for gl, hl, graphl in zip(g, x, graph):
            hn = self.gcn_layer(norm_g(gl), hl)
            layer_norm = nn.LayerNorm(normalized_shape=hn.size(), elementwise_affine=False)
            hn = layer_norm(hn)
            hn = F.dropout(hn, p=self.args.drop_n)

            g, h, idx, subs = Subgraph().subgraphs(gl, hn)
            selected_subs, sub_representations = self.select_layers(hn, subs)
            self.node_search_layers = NodeMixedOp(selected_subs, graphl, node_alphas)
            sub_representations = sub_representations.cuda()
            b = self.node_search_layers(hn, sub_representations)
            bs.append(b)
            new_subs, _ = NODE_OPS[gene_node](selected_subs, graphl)(hn, sub_representations)
            new_subs_n.append(new_subs)
            hbatch.append(hn)

        B = torch.zeros(len(bs), b.shape[1]).cuda()
        for i in range(B.shape[0]):
            B[i] = bs[i].squeeze(0)
        idx = np.random.permutation(len(bs))
        shuf_fts = B[idx]
        ret = self.disc(original, B, shuf_fts, s_bias1=None, s_bias2=None)
        return ret, gene_node, new_subs_n, hbatch

    def _initialize_alphas(self):
        num_node_ops = len(NODE_PRIMITIVES)
        if self.args.model_type == 'darts':
            self.log_node_alphas = Variable(1e-3 * torch.randn(1, num_node_ops).cuda(),
                                            requires_grad=True)
            
        else:
            self.log_node_alphas = Variable(
                torch.ones(1, num_node_ops).normal_(self.args.loc_mean, self.args.loc_std).cuda(),
                requires_grad=True)

        self._arch_parameters = [self.log_node_alphas]

    def arch_parameters(self):
        return self._arch_parameters

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

class EdgeSearch(nn.Module):
    def __init__(self, args, x, graph, subs):
        super(EdgeSearch, self).__init__()
        self.args = args
        self.edge_search_layers = nn.ModuleList()
        self.disc = Discriminator(self.args.l_dim)
        self._initialize_alphas()
        if self.args.model_type == 'darts':
            edge_alphas = F.softmax(self.log_edge_alphas, dim=-1)
        else:
            edge_alphas = self._get_categ_mask(self.log_edge_alphas)
        self.edge_alphas = edge_alphas
        for hl, graphl, sub in zip(x, graph, subs):
            self.edge_search_layers.append(EdgeMixedOp(sub, graphl, self.edge_alphas, hl, self.args))

    def _get_categ_mask(self, alpha):
        log_alpha = alpha
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self.args.temp)
        return one_hot



    def forward(self, original, x, g, graph, subs):
        edge_subs = []
        cs = []
        
        edge_indices = torch.argmax(self.edge_alphas, dim=-1)
        gene_edge = EDGE_PRIMITIVES[edge_indices]
        bat = 0
        for gl, hl, graphl, sub in zip(g, x, graph, subs):
            c = self.edge_search_layers[bat](hl, graphl, self.args)
            cs.append(c)
            new_edge_subs, new_sub_representations = EDGE_OPS[gene_edge](sub, graphl, hl, self.args)(hl, graphl, self.args)
            edge_subs.append(new_edge_subs)
            bat += 1

        C = torch.zeros(len(cs),c.shape[1]).cuda()
        for i in range(C.shape[0]):
            C[i] = cs[i].squeeze(0)
        
        idx = np.random.permutation(len(cs))
        shuf_fts = C[idx]

        ret = self.disc(original, C, shuf_fts, s_bias1=None, s_bias2=None)
        return ret, gene_edge, edge_subs, x


    def _initialize_alphas(self):
        num_edge_ops = len(EDGE_PRIMITIVES)

        if self.args.model_type == 'darts':
            self.log_edge_alphas = Variable(1e-3 * torch.randn(1, num_edge_ops).cuda(),
                                            requires_grad=True)
        else:
            self.log_edge_alphas = Variable(
                torch.ones(1, num_edge_ops).normal_(self.args.loc_mean, self.args.loc_std).cuda(),
                requires_grad=True)

        self._arch_parameters = [self.log_edge_alphas]

    def arch_parameters(self):
        return self._arch_parameters


def to_cuda(gs):
    if torch.cuda.is_available():
        if type(gs) == list:
            return [g.cuda() for g in gs]
        return gs.cuda()
    return gs


