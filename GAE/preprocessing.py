import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):  # 是否为csr_matrix类型
        sparse_mx = sparse_mx.tocoo()  # 实现csc矩阵转换为coo矩阵
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组，堆叠的数组需要具有相同的维度，transpose()作用是转置
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)  # csr_matrix转成coo_matrix
    adj_ = adj + sp.eye(adj.shape[0])  # S=A+I  #注意adj_的类型为csr_matrix
    rowsum = np.array(adj_.sum(1))  # rowsum的shape=(节点数,1)，对于cora数据集来说就是(2078,1)，sum(1)求每一行的和
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # 计算D^{-0.5}
    # p.diags：提取输入矩阵(大小为m×n)的所有非零对角列。输出的大小为 min(m,n)×p，其中p表示输入矩阵的p个非零对角列
    # numpy.power()：用于数组元素求n次方
    # flatten()：返回一个折叠成一维的数组。
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # adj_.dot(degree_mat_inv_sqrt)得到 SD^{-0.5}
    # adj_.dot(degree_mat_inv_sqrt).transpose()得到(D^{-0.5})^{T}S^{T}=D^{-0.5}S，因为D和S都是对称矩阵
    # adj_normalized即为D^{-0.5}SD^{-0.5}
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0
    # assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。
    # todense()方法将稀疏矩阵b转换成稠密矩阵c

    adj_triu = sp.triu(adj)  # 取出稀疏矩阵的上三角部分的非零元素，返回的是coo_matrix类型
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    # 取除去节点自环的所有边（注意，由于adj_tuple仅包含原始邻接矩阵上三角的边，所以edges中的边虽然只记录了边<src,dis>，而不冗余记录边<dis,src>），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    edges_all = sparse_to_tuple(adj)[0]
    # 取原始graph中的所有边，shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    num_test = int(np.floor(edges.shape[0] / 10.))  # 划分测试集
    # np.floor返回数字的下舍整数
    num_val = int(np.floor(edges.shape[0] / 20.))  # 划分验证集
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)  # 打乱all_edge_idx的顺序
    val_edge_idx = all_edge_idx[:num_val]  # 划分验证集
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]  # 划分测试集
    test_edges = edges[test_edge_idx]  # edges是除去节点自环的所有边（因为数据集中的边都是无向的，edges只是存储了<src,dis>,没有存储<dis,src>，因为没必要浪费内存），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    # np.vstack():在竖直方向上堆叠，np.hstack():在水平方向上平铺。
    # np.hstack([test_edge_idx, val_edge_idx])将两个list水平方向拼接成一维数组
    # np.delete的参数axis=0，表示删除多行，删除的行号由第一个参数确定

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        # np.round返回浮点数x的四舍五入值，第二参数是保留的小数的位数
        # b[:, None]使b从shape=(边数,2)变为shape=(边数,1,2)，而a是长度为2的list，a - b[:, None]触发numpy的广播机制
        # np.all()判断给定轴向上的所有元素是否都为True，axis=-1（此时等同于axis=2）表示3维数组最里层的2维数组的每一行的元素是否都为True
        return np.any(rows_close)
        # np.any()判断给定轴向上是否有一个元素为True,现在不设置axis参数则是判断所有元素中是否有一个True，有一个就返回True。
        # rows_close的shape=(边数,1)
        # 至此，可以知道，ismember( )方法用于判断随机生成的<a,b>这条边是否是已经真实存在的边，如果是，则返回True，否则返回False

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])  # 生成负样本
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)  # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。所以，这里是想要edges_all不含有test_edges_false，否则抛异常
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # 重建出用于训练阶段的邻接矩阵
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # #注意：这些边列表只包含一个方向的边（adj_train是矩阵，不是edge lists）
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
