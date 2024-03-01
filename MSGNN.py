import torch.nn as nn
from dataset import SubgraphData
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from node_search import get_args
from data_loader import FileLoader, SubFileloader, G_data
import time
import random
from node_search import NodeSearch,EdgeSearch
from functools import reduce
import operator
from ops import GCN, Subgraph, norm_g
import torch.utils.data as Data
import networkx as nx
import configparser
import argparse
import torch.nn.functional as F
import cluster_step
import time


######################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MUTAG", dest='dataset')
args = parser.parse_known_args()[0]
######################################################################


#os.chdir('dataset\{}'.format(args.dataset))
config = configparser.ConfigParser()
config.read('.config')
getConfig = lambda x: config.get('config', x)
ds = getConfig('ds')
cate = int(getConfig('cate'))
class_size = int(getConfig('class_size'))


def to_cuda(gs):
    if torch.cuda.is_available():
        if type(gs) == list:
            return [g.cuda() for g in gs]
        return gs.cuda()
    return gs


class MSGNN(nn.Module):
    def __init__(self, gall, hall, ks, dim, args):
        super(MSGNN, self).__init__()
        self.ks = ks
        self.args = args
        self.pools = nn.ModuleList()
        self.l_n = len(ks)
        self.out_l_1 = nn.Linear(self.args.l_dim*self.l_n, self.args.h_dim)
        self.out_l_2 = nn.Linear(self.args.h_dim, class_size)
        self.out_drop = nn.Dropout(p=args.drop_c)
        self.c_act = getattr(nn, args.act_c)()
        self.gall = gall
        self.hall = hall
        self.gcn_layer = GCN(cate, self.args.l_dim, F.elu, self.args.drop_n)
        for i in range(self.l_n):
            self.pools.append(Pool(ks[i], dim, args))

    def forward(self, data, len_data, labels):
        labels = torch.Tensor(labels)
        labels = labels.to(torch.int64).cuda()
        gall = self.gall
        hall = self.hall
        h_tensor_all = []
        o_hs = []
        for gl, hl in zip(gall, hall):
            gl, hl = map(to_cuda, [gl, hl])
            gl = norm_g(gl)
            hn = self.gcn_layer(gl, hl)
            h_tensor_all.append(hn)
            h_sum = torch.sum(hn, 0)
            o_hs.append(h_sum)

        original = torch.zeros(len(o_hs), h_sum.shape[0]).cuda()
        for i in range(original.shape[0]):
            original[i] = o_hs[i]

        h_tensor = torch.zeros((len_data, self.args.l_dim * self.l_n)).cuda()
        node_sub = []
        for i in range(self.l_n):
            node_geno, node_sub, gall, hall = self.pools[i](data, len_data, original, gall, hall, NodeSearch, node_sub, name='Node')
            # print('node_geno', node_geno)
            edge_geno, edge_sub, gall, hall = self.pools[i](data, len_data, original, gall, hall, EdgeSearch, node_sub, name='Edge')
            # print('edge_geno', edge_geno)

            for hidx in range(len(hall)):
                h_tensor[hidx, i * self.args.l_dim: (i+1) * self.args.l_dim] = torch.sum(hall[hidx], 0)

        #cluster
        clu = []
        for i in range(len(edge_sub)):
            graph = nx.from_numpy_array(self.gall[i].numpy())
            cl, remain_nodes = cluster_step.cluster(edge_sub[i], graph)
            if list(set(cl)) == [0]:
                r_cl = h_tensor_all[i][remain_nodes]
                r_cl = torch.mean(r_cl,dim=0)
                h_tensor[i, (self.l_n - 1) * self.args.l_dim: self.l_n * self.args.l_dim] = h_tensor[i, (self.l_n - 1) * self.args.l_dim: self.l_n * self.args.l_dim] + r_cl
            elif list(set(cl)) == [] or [-1]:
                h_tensor = h_tensor
            else:
                h_add = torch.zeros(1, self.args.l_dim)
                dic = {k:[] for k in list(set(cl))}
                for k in list(set(cl)):
                    k_index = [j for j, v in enumerate(cl) if v == k]
                    dic[k].append(remain_nodes[ind] for ind in k_index)
                    if k != -1:
                        h_add = h_add + torch.mean(self.h_tensor_all[i][dic[k]], dim=0)
                h_tensor[i, (self.l_n - 1) * self.args.l_dim: self.l_n * self.args.l_dim] = h_tensor[i,(self.l_n - 1) * self.args.l_dim: self.l_n * self.args.l_dim] + h_add
            clu.append(cl)
        logits = self.classify(h_tensor)
        return self.metric(logits, labels)

    def classify(self, h):
        h = self.out_drop(h)
        h = self.out_l_1(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        return F.log_softmax(h, dim=1)

    def metric(self, logits, labels):
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        return loss, acc


class Pool(nn.Module):
    def __init__(self, k, dim, args):
        super(Pool, self).__init__()
        self.k = k
        self.proj = nn.Linear(dim, 1)
        self.in_dim = cate
        self.args = args

    def forward(self, data, len_data, original, gall, hall, model, node_subs, name):

        for epoch in range(self.args.MIepochs):
            print('MIepoch:', epoch)

            for i in range(self.args.w_update_epoch):
                subs = []
                adjall = []
                emball = []
                batch_len = 0
                losses = []
                for batch in tqdm(data, unit='b'):
                    cur_len, subgraphs, hs, gs, graphs, labels = batch 
                    gs, hs = map(to_cuda, [gs, hs])
                    batch_len = batch_len + len(gs)

                    if name == 'Node':

                        model1 = model(self.k, self.in_dim, self.args.l_dim, self.args).cuda()

                        optimizer = optim.Adam(model1.parameters(), lr=self.args.learning_rate, amsgrad=True,
                                               weight_decay=self.args.weight_decay)
                        arch_optimizer = optim.Adam(model1.arch_parameters(), lr=self.args.arch_learning_rate,
                                                    amsgrad=True, weight_decay=self.args.arch_weight_decay)
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.args.num_epochs),
                                                                         eta_min=self.args.learning_rate_min)  # 学习率衰减，按照余弦波形的衰减周期来更新学习率
                        scheduler_arch = optim.lr_scheduler.CosineAnnealingLR(arch_optimizer,
                                                                              float(self.args.num_epochs),
                                                                              eta_min=self.args.arch_learning_rate_min)
                        model1.train()
                        optimizer.zero_grad() 
                        arch_optimizer.zero_grad()

                        logits, geno, new_subs, hbatch = model1(original[(batch_len - len(gs)):batch_len], hs, gs, graphs)
                        subs.extend(new_subs)

                        for index in range(len(new_subs)):
                            emball.append(hbatch[index])
                            adjall.append(gs[index])
                        
                    if name == 'Edge':

                        model1 = model(self.args, hall[(batch_len - len(gs)):batch_len], graphs, node_subs[(batch_len - len(gs)):batch_len]).cuda()

                        optimizer = optim.Adam(model1.parameters(), lr=self.args.learning_rate, amsgrad=True,
                                               weight_decay=self.args.weight_decay)
                        arch_optimizer = optim.Adam(model1.arch_parameters(), lr=self.args.arch_learning_rate,
                                                    amsgrad=True, weight_decay=self.args.arch_weight_decay)
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.args.num_epochs),
                                                                         eta_min=self.args.learning_rate_min)  # 学习率衰减，按照余弦波形的衰减周期来更新学习率
                        scheduler_arch = optim.lr_scheduler.CosineAnnealingLR(arch_optimizer,
                                                                              float(self.args.num_epochs),
                                                                              eta_min=self.args.arch_learning_rate_min)
                        model1.train()
                        optimizer.zero_grad()  
                        arch_optimizer.zero_grad()

                        logits, geno, new_subs, hbatch = model1(original[(batch_len - len(gs)):batch_len], hall[(batch_len - len(gs)):batch_len], gs, graphs, node_subs[(batch_len - len(gs)):batch_len])
                        subs.extend(new_subs)

                        for index in range(len(new_subs)):
                            pool_graph = list(set(reduce(operator.add, new_subs[index])))
                            adj = gs[index][pool_graph,:]
                            adj = adj[:, pool_graph]
                            adjall.append(adj)
                            emball.append(hbatch[index][pool_graph,:])

                    lbl_1 = torch.ones(cur_len, 1)
                    lbl_2 = torch.zeros(cur_len, 1)
                    lbl = torch.cat((lbl_1, lbl_2), 1).cuda()
                    b_xent = nn.BCEWithLogitsLoss()
                    loss = b_xent(logits, lbl)

                    losses.append(loss*cur_len)

                    lossMI = loss.detach_().requires_grad_(True)
                    arch_optimizer.zero_grad()
                    lossMI.backward(retain_graph=True)
                    optimizer.step()  
                    scheduler.step()
                loss = sum(losses) / len_data
                print('MIloss:{:.08f}'.format(loss.item()))
                optimizer.zero_grad()
                arch_optimizer.step()
                scheduler_arch.step()
                gall = adjall
                halls = emball

        return geno, subs, gall, halls


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = get_args()
    print(args)
    set_random(args.seed)
    start = time.time()
    G_data, g_list = FileLoader(args).load_data()
    print('load data using ------>', time.time() - start)
    gall = []
    hall = []
    labels = []
    for g in g_list:
        gall.append(g.A)
        hall.append(g.feas)
        labels.append(g.label)
    node_subs = []
    Sub_Data, g_lists = SubFileloader(gall, hall, labels, node_subs).load_subdate()
    Sub_Data.use_fold_data(fold_idx=1)
    print('#train: %d, #test: %d' % (len(Sub_Data.train_gs), len(Sub_Data.test_gs)))
    len_train = len(Sub_Data.train_gs)
    len_test = len(Sub_Data.test_gs)
    train_data = SubgraphData(Sub_Data.train_gs)
    test_data = SubgraphData(Sub_Data.test_gs)
    train_label = []
    for i in range(len_train):
        train_label.append(train_data[i][4])
    test_label = []
    for i in range(len_test):
        test_label.append(test_data[i][4])
    train_d = train_data.loader(args.batch, False)
    test_d = test_data.loader(args.batch, False)

    net = MSGNN(gall, hall, args.ks, args.l_dim, args)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    # h_tensor, gall = net(gall, hall)
    train_str = 'Train epoch %d: loss %.5f acc %.5f'
    test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f'
    max_acc = 0.0
    for eid in range(args.num_epochs):
        since = time.time()
        net.train()
        loss, acc = net(train_d, len_train, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_elapsed = time.time() - since
        print(train_str % (eid, loss.item(), acc.item()))
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))
        with torch.no_grad():
            net.eval()
            loss, acc = net(test_d, len_test, test_label)
        max_acc = max(max_acc, acc)
        print(test_str % (eid, loss, acc, max_acc))


if __name__ == "__main__":
    newsubs = main()




