import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args

class VGAE(nn.Module):
    def __init__(self, adj):
        super(VGAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        # lambda是匿名函数，冒号左边是参数，多个参数用逗号隔开，右边是表达式
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        # 这里使用torch.exp是因为论文中log(sigma)=GCN_{sigma}(X,A)，torch.exp(self.logstd)即torch.exp(log(sigma))得到的是sigma；另外还有mu=GCN_{mu}(X,A).
        # 由于每个节点向量经过GCN后都有且仅有一个节点向量表示，所以呢，方差的对数log(sigma)和节点向量表示的均值mu分别是节点经过GCN_{sigma}(X,A)和GCN_{mu}(X,A)后得到的向量表示本身。
        # 从N(mu,sigma^2)中采样一个样本Z相当于在N(0,1)中采样一个xi，然后Z = mu + xi×sigma
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim)
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		#torch.mm(a, b)是矩阵a和b矩阵相乘
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

class GAE(nn.Module):
	def __init__(self,adj):
		super(GAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred