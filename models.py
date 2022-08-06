import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math

from layers import AvgReadout, Discriminator, GCN

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.fc = nn.Linear(n_in, n_h)
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.act = nn.PReLU()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, x_1, x_2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(x_1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.gcn(x_2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        # h_1 = self.sigm(self.fc(seq))
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print(f'in_feat: {in_features}, out_feat: {out_features}')
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output

class GCN_Classifier(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

def get_model(model_opt, nfeat, nclass, nhid=0, dropout=0, cuda=True):
    if model_opt == "GCN":
        model = GCN_Classifier(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass)
    elif model_opt == 'distance_based':
        model = distance_based(nfeat=nfeat, nembed=nhid, nclass=nclass)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model


class distance_based(nn.Module):
    """
    distance_based classifier.
    The input feature should be DGI features.
    """
    def __init__(self, nfeat, nembed, nclass):
        super(distance_based, self).__init__()
        self.nfeat = nfeat
        self.nembed = nembed
        self.nclass = nclass
        self.W = nn.Linear(nfeat, nembed)
        self.class_embed = nn.Embedding(nclass, nembed)

    def forward(self, x):
        u = self.W(x)
        num_nodes = u.size(0)
        u = u.view(num_nodes, -1, self.nembed)
        class_embed = self.class_embed.weight.view(-1, self.nclass, self.nembed)
        distances = torch.norm(u - class_embed, dim=-1)
        return distances

    def new_features(self, x):
        u = self.W(x)
        return u.detach()