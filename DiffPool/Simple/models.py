import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
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
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' +str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=-1)


class Pooling(nn.Module):
    def __init__(self, input, output):
        super(Pooling, self).__init__()
        self.gc1 = GraphConvolution(input, output)

    def forward(self, x, adj):
        z = F.relu(self.gc1(x, adj))
        s = F.log_softmax(z, dim=-1)
        X = torch.matmul(s.transpose(1, 2), z)
        A = s.transpose(1, 2) @ adj @ s
        return X, A

class GCNpooling(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNpooling, self).__init__()
        self.pooling1 = Pooling(nfeat, nhid)
        self.gc = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1, adj1 = self.pooling1(x, adj)
        z = F.relu(self.gc(x1, adj1))
        z = F.dropout(z, self.dropout, training=self.training)
        #out = F.log_softmax(z, dim=-1)
        out, _ = torch.max(z, dim=1)
        return out

    def loss(self, pred, label):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = F.cross_entropy(pred, label, reduction='mean')
        # max_num_nodes = adj.size()[1]
        # pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
        # tmp = pred_adj0
        # pred_adj = pred_adj0
        # for adj_pow in range(adj_hop - 1):
        #     tmp = tmp @ pred_adj0
        #     pred_adj = pred_adj + tmp
        # pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype))  # .cuda()
        # self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
        # if batch_num_nodes is None:
        #     num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
        #     print('Warning: calculating link pred loss without masking')
        # else:
        #     num_entries = np.sum(batch_num_nodes * batch_num_nodes)
        #     embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        #     adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
        #     self.link_loss[(1 - adj_mask).bool()] = 0.0
        #
        # self.link_loss = torch.sum(self.link_loss) / float(num_entries)
        # return loss + self.link_loss
        return loss
