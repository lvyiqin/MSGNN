import torch
import torch.nn as nn
import argparse


parser = argparse.ArgumentParser("pas-train-search")
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5.979931414632729e-05, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
args = parser.parse_args()


class Top_k_Subs(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Top_k_Subs, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(self.in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, x, subs):
        i = 0
        sub_representations = torch.randn(len(subs), self.in_dim)
        for sub in subs:
            sub_representations[i] = torch.sum(x[sub, :], 0)
            i += 1
        Z = self.drop(sub_representations)
        Z = Z.cuda()
        weights = self.proj(Z).squeeze()
        if len(subs) >= 2:
            scores = self.sigmoid(weights)
            values, idx = torch.topk(scores, max(2, int(self.k * len(subs))))
            selected_subs = [subs[idx[i]] for i in range(idx.shape[0])]
            selected_sub_representations = sub_representations[idx]
        else:
            selected_subs = subs
            selected_sub_representations = sub_representations
        return selected_subs, selected_sub_representations





