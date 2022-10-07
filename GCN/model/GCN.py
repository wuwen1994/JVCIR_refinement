import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # [13359, 5]   *  w0: [5, 32] =   [13359, 32]
        output = torch.spmm(adj, support)  # 邻接矩阵 [13359, 13359] *  [13359, 32]  =  [13359, 32]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        """
        :param nfeat: 5
        :param nhid: 32
        :param nclass: 1
        :param dropout: 0.5
        """
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  # 输入为5通道 输出为32通道
        self.gc2 = GraphConvolution(nhid, nclass)  # 输入为32通道，输出为1通道
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # x: [13359, 5]   # [13359, 13359]的邻接矩阵 * w + 对角矩阵 # [13359, 32]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)   # [13359, 32]   # [13359, 13359]  # [13359, 1]
        return torch.sigmoid(x)[:,0]
