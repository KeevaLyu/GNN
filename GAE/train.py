import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time

import args
from input_data import load_data
from preprocessing import *
import model

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

adj, features = load_data(args.dataset)

# 存储原始邻接矩阵(无对角线条目)以备后用
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
# Scipy的dia_matrix函数见1.其中offsets数组中0表示对角线，-1表示对角线下面，正数表示对角线上面
# np.newaxis的作用是增加一个维度。[np.newaxis，：]是在np.newaxis这里增加1维。这样改变维度的作用往往是将一维的数据转变成一个矩阵
#diagonal()是获得矩阵对角线
#adj_orig.diagonal()[np.newaxis, :], [0]代码意思是先将对角线提取出来然后增加一维变为矩阵，方便后续计算
adj_orig.eliminate_zeros()
#eliminite_zeros() 存储去掉0元素，返回的是稀疏存储

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train #用于训练的邻接矩阵，类型为csr_matrix

# Some preprocessing
adj_norm = preprocess_graph(adj)
#返回D^{-0.5}SD^{-0.5}的coords(坐标), data, shape，其中S=A+I
num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
#features的类型原为lil_matrix，sparse_to_tuple返回features的coords, data, shape
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#注意，adj的每个元素非1即0。pos_weight是用于训练的邻接矩阵中负样本边（既不存在的边）和正样本边的倍数（即比值），这个数值在二分类交叉熵损失函数中用到，
#如果正样本边所占的比例和负样本边所占比例失衡，比如正样本边很多，负样本边很少，那么在求loss的时候可以提供weight参数，将正样本边的weight设置小一点，负样本边的weight设置大一点，
#此时能够很好的平衡两类在loss中的占比，任务效果可以得到进一步提升
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0]) #adj_train是用于训练的邻接矩阵，类型为csr_matrix
adj_label = sparse_to_tuple(adj_label)


adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),  #其中adj_norm是D^{-0.5}SD^{-0.5}的coords, data, shape
                            torch.FloatTensor(adj_norm[1]),
                            torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                            torch.FloatTensor(adj_label[1]),
                            torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                            torch.FloatTensor(features[1]),
                            torch.Size(features[2]))

weight_mask = adj_label.to_dense().view(-1) == 1
# view的参数-1 表示做自适应性调整，如果参数只有一个参数-1,则表示将Tensor变成一维张量。
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight
#用于在binary_cross_entropy中设置正样本边的weight。负样本边的weight都为1，正样本边的weight都为pos_weight


# init model and optimizer
model = getattr(model, args.model)(adj_norm)
#getattr() 函数用于返回一个对象属性值。
optimizer = Adam(model.parameters(), lr=args.learning_rate)

def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        # item()取出单元素张量的元素值并返回该值，保持原元素类型不变，从而能够保留原来的精度。所以在求loss,以及accuracy rate的时候一般用item()
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def get_acc(adj_label, adj_rec):
    labels_all = adj_label.to_dense().view(-1).long()   #long()将数字或字符串转换为一个长整型
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

# train model
for epoch in range(args.num_epoch):
    t = time.time()

    A_pred = model(features)    #得到的A_pred每个元素不再是非1即0
    optimizer.zero_grad()
    loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
    if args.model == 'VGAE':
        kl_divergence = 0.5 / A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
        # kl_divergence就是正态分布的KL散度，即n个(0.5*(1+log(sigma^2)-mu^2-sigma^2))的和，n为图中节点的数量，也就是这里的A_pred.size(0)
        # 2*model.logstd即为2*log(sigma)，根据运算法则，log(sigma^2)=2*log(sigma)；model.mean**2即为mu^2；torch.exp(model.logstd)**2即为sigma^2
        # 1+log(sigma^2)-mu^2-sigma^2
        # sum(1)表示矩阵每一行内元素求和
        loss -= kl_divergence

    loss.backward()
    optimizer.step()

    train_acc = get_acc(adj_label, A_pred)

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))


test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))
