import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, spGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        """
           参数1 ：nfeat   输入层数量
           参数2： nhid    输出特征数量
           参数3： nclass  分类个数
           参数4： dropout dropout概率
           参数5： alpha  激活函数的斜率
           参数6： nheads 多头部分

        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # 根据多头部分给定的数量声明attention的数量
        # 将多头的各个attention作为子模块添加到当前模块中
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 最后一个attention层，输出的是分类
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1) #h_i..k
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1) #h

class spGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(spGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [spGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = spGraphAttentionLayer(nhid*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)