import torch
import time
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
from GCN.model.GCN import GCN
from GCN.config import cashfile

from utils import supervised_loss, normalize, normalize_adj, sparse_mx_to_torch_sparse_tensor, accuracy


def load_data():
    #print("--------加载图数据--------")
    val_portion = 0.2

    graph_path = cashfile + "graph.npy"
    weights_path = cashfile + "graph_weights.npy"
    features_path = cashfile + "graph_node_features.npy"
    labels_path = cashfile + "graph_ground_truth.npy"
    mask_path = cashfile + "unc_mask.npy"  # 不参与训练但要参与测试
    y_test_path = cashfile + "reference_graph.npy"  # GT，有些任务中，它不一定真实

    graph = np.load(graph_path)
    weights = np.load(weights_path)
    features = np.load(features_path)
    y_test = np.load(y_test_path)  # GT，有些任务中，它不一定真实
    test_mask = np.load(mask_path)  # 不参与训练但要参与测试
    full_mask = 1 - test_mask  # 参与训练的样本

    labels = np.load(labels_path)  # ROI区域所有节点的CNN初始预测值, 其中可靠样本拿走标签，不可靠样本标签依赖后续的传播
    num_nodes = labels.shape[0]  # 节点的个数
    adj = sp.coo_matrix((weights, (graph[:, 0], graph[:, 1])), shape=(num_nodes, num_nodes))  # 新建一个矩阵，将权重放到相应的位置
    features = sp.coo_matrix(features)  # numpy.ndarray -> tuple
    working_nodes = np.where(full_mask != 0)[0]  # 可靠的节点 (12594,)
    random_arr = np.random.uniform(low=0, high=1, size=working_nodes.shape)  # 生成随机数组(12594,)

    features = normalize(features)  # (13359, 5) 特征归一化    特征矩阵
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # 权重矩阵  加上 对角 矩阵?

    # 确定训练集和验证集节点的ID
    idx_train = working_nodes[random_arr > val_portion]
    idx_val = working_nodes[random_arr <= val_portion]
    # 测试集
    idx_test = np.where(test_mask != 0)

    # print("节点数量: {}".format(num_nodes))
    # print("不可靠节点数量: {}".format(int(np.sum(test_mask))))
    # print("可靠节点数量: {}".format(int(np.sum(full_mask))))
    # print("可靠的阴影节点数量: {}".format(np.sum(labels[np.where(full_mask != 0)[0]] == 1)))
    # print("可靠的非阴影节点数量: {}".format(np.sum(labels[np.where(full_mask != 0)[0]] == 0)))

    # 转化数据类型为Tensor, 准备训练
    features = torch.FloatTensor(np.array(features.todense()))  # sparse.matrix -> Tensor  [13359, 5]
    labels = torch.FloatTensor(labels)  # ndarray -> Tensor  [13359]
    y_test = torch.FloatTensor(y_test[:, 0])  # ndarray -> Tensor  [13359]
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # sparse.matrix -> Tensor [13359, 13359]

    idx_train = torch.LongTensor(idx_train)  # ndarray ->  Tensor   [10047]
    idx_val = torch.LongTensor(idx_val)  # ndarray ->  Tensor  [2547]
    idx_test = torch.LongTensor(idx_test[0])  # tuple -> Tensor  [765]

    return adj, features, labels, y_test, idx_train, idx_val, idx_test


def train(model, optimizer, epoch, adj, features, labels, idx_train, idx_val, idx_test):
    t = time.time()
    # 训练
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    sup_loss = supervised_loss(output[idx_train], labels[idx_train])
    # one_hot_loss = entropy_loss(output[idx_test])
    loss_train = sup_loss
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # 测试
    model.eval()
    output = model(features, adj)
    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = supervised_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def GCN_training(epochs, dropout):
    # 训练设置
    seed = 42
    lr = 1e-2
    weight_decay = 1e-5
    hidden = 32
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # 加载数据
    adj, features, labels, y_test, idx_train, idx_val, idx_test = load_data()
    # adj:   权重矩阵+对角矩阵
    # features: 特征矩阵
    # labels: CNN初始预测->作为GCN中的label信息的提供者
    # y_test: real GT
    # idx_train, idx_val, idx_test: 训练集，验证集，和测试集

    # 加载模型

    model = GCN(nfeat=features.shape[1],  # 输入5维度
                nhid=hidden,  # 隐藏层32维度
                nclass=1,  # 输出为1维
                dropout=dropout)  # dropout 0.5

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    # 导入cuda
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    y_test = y_test.cuda()

    # 模型训练
    torch.set_grad_enabled(True)
    t_total = time.time()
    model.eval()
    # print("------- Training GCN")
    for epoch in range(epochs):
        train(model, optimizer, epoch, adj, features, labels, idx_train, idx_val, idx_test)
    # print("Optimization Finished!")
    # print("Total time: {:.4f}s".format(time.time() - t_total))

    return model, adj, features, y_test, idx_test
