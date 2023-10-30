import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    带有attention计算的网络层

    参数：in_features 输入节点的特征数F
    参数：out_features 输出的节点的特征数F'
    参数：dropout
    参数：alpha LeakyRelu激活函数的斜率
    参数：concat
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.alpha = alpha  # 激活斜率 (LeakyReLU)的激活斜率
        self.concat = concat  # 用来判断是不是最后一个attention # if this layer is not last layer,

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # 建立一个w权重，用于对特征数F进行线性变化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 对权重矩阵进行初始化 服从均匀分布的Glorot初始化器
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # 计算函数α，输入是上一层两个输出的拼接，输出的是e_ij，a的size为(2*F',1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        参数h：表示输入的各个节点的特征矩阵
        参数adj ：表示邻接矩阵
        '''
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # 线性变化特征的过程,Wh的size为(N,F')，N表示节点的数量，F‘表示输出的节点的特征的数量
        e = self._prepare_attentional_mechanism_input(Wh) # e = aW(h_i||h_j), shape = (N, N, 2 * out_features)

        zero_vec = -9e15 * torch.ones_like(e)  # 生成一个矩阵，size为(N,N)
        attention = torch.where(adj > 0, e, zero_vec) # 对于邻接矩阵中的元素，>0说明两种之间有边连接，就用e中的权值，否则表示没有边连接，就用一个默认值来表示
        attention = F.softmax(attention, dim=1) # 做一个softmax，生成贡献度权重
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh) # h_i = attention * Wh

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        '''
        	#下面是self-attention input ，构建自我的特征矩阵
            #matmul的size为(N,1)表示e_ij对应的数值
            #e的size为(N,N)，每一行表示一个节点，其他各个节点对该行的贡献度
            '''
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # 矩阵乘法
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    # 打印输出类名称，输入特征数量，输出特征数量
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer.对稀疏区域的反向传播函数"""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class spGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(spGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W) # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t() # edge_h: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze())) # edge_e: E
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv)) # e_rowsum: N x 1
        edge_e = self.dropout(edge_e) # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h) # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum) # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
