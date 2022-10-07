import numpy as np
import scipy.sparse as sp
from utils import map_pixel_nodes
from GCN.config import cashfile


class Weighting():
    def __init__(self):
        super(Weighting, self).__init__()
        self.description = "l*div + e(int) + e(pos)"
        self.weights1 = []
        self.weights2 = []
        self.weights3 = []

    def weights_for(self, idx1, idx2, args):
        """
        :param idx1: 节点1的坐标
        :param idx2: 节点2的坐标
        :param args:
        :return:
        """
        prob1 = args["probability"][idx1]  # 节点1的预测期望值
        prob2 = args["probability"][idx2]  # 节点2的预测期望值
        int1 = args["shadow"][:, idx1[0], idx1[1]]  # 节点1的三通道像素值
        int2 = args["shadow"][:, idx2[0], idx2[1]]  # 节点2的三通道像素值
        _, ny, nx = args["shadow"].shape  # limit区域的高和宽
        dim_array = np.array([ny, nx], dtype=np.float32)  # 新建一个与limit长宽一致的array
        pos1 = np.array(idx1, dtype=np.float32) / dim_array  # 归一化节点1的position
        pos2 = np.array(idx2, dtype=np.float32) / dim_array  # 归一化节点2的position
        #   计算两个节点之间的相似性：节点的期望预测值，RGB三通道的值，节点的空间位置
        # 因为数据提前做了标准化处理，所以以下算出来的结果均直接为相似性，不需要再除以方差，然后计算以e为底的对数值
        int_diff = int1 - int2  # 颜色差异
        pos_diff = pos1 - pos2  # 空间位置差异
        intensity = np.sum(int_diff * int_diff)  # 颜色的L2 distance
        space = np.sum(pos_diff * pos_diff)  # 空间位置的L2 distance
        p = prob1 - prob2
        delta = 1.0e-15
        lambd = p * (np.log2(prob1 / (prob2 + delta) + delta) - np.log2((1 - prob1) / ((1 - prob2) + delta) + delta))

        self.weights1.append(lambd)  # 保存该节点的期望相似性
        self.weights2.append(intensity)  # 保存该节点的颜色相似性
        self.weights3.append(space)  # 保存该节点的空间位置相似性

    def post_process(self, args=None):
        self.weights1 = np.asarray(self.weights1, dtype=np.float32)  # 266896
        self.weights2 = np.asarray(self.weights2, dtype=np.float32)  # 266896
        self.weights3 = np.asarray(self.weights3, dtype=np.float32)  # 266896
        num_nodes = args["num_nodes"]  # ROI区域节点个数  13373
        ne = float(self.weights1.shape[0])  # number of edges 边的数量

        muw2 = self.weights2.sum() / ne  # weight2的平均值
        muw3 = self.weights3.sum() / ne  # weight3的平均值

        sig2 = 2 * np.sum((self.weights2 - muw2) ** 2) / ne  # 权重2的方差
        sig3 = 2 * np.sum((self.weights3 - muw3) ** 2) / ne  # 权重3的方差

        self.weights2 = np.exp(-self.weights2 / sig2)
        self.weights3 = np.exp(-self.weights3 / sig3)

        # h: args["edges"][:, 0]   所有边的起始节点ID  266896
        # w: args["edges"][:, 1]   所有边的目标节点ID  266896
        # 为ROI上的所有节点，构建一个正方形的权重矩阵w
        # 把[h,w]处的权重weights的值依次放到矩阵w中去,其他值填充为0
        w1 = sp.coo_matrix((self.weights1, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        w2 = sp.coo_matrix((self.weights2, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        w3 = sp.coo_matrix((self.weights3, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))

        self.weights = 0.5 * w1 + w2 + w3

    def get_weights(self):
        return self.weights


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def connect_n6_krandom(ref, pixel_node, node_pixel, working_nodes, k_random, weighting, args):
    """
    :param ref: 不要Dropout时，ROI上的初始预测值  [144, 125]
    :param pixel_node: 字典 例如 {(0, 51): 0, (0, 52): 1, (0, 53): 2, (0, 54): 3, (0, 55): 4}
    :param node_pixel: [[0 51] [0 52].......]
    :param working_nodes:  ROI区域
    :param k_random:  16个全局的随机
    :param weighting: Weight实例对象
    :param args: 参数
    :return:
    """
    edges = []  # 二维矩阵，表示ID*和ID*的节点之间连接在一起
    labels = []  # CNN prediction‘s label
    num_nodes = node_pixel.shape[0]  # 图节点个数
    tabu_list = {}  # A list to avoid duplicated elements in the adjacency matrix. 邻接矩阵
    nodes_complete = {}  # A list counting how many neighbors a node already has.  度矩阵
    valid_nodes = np.array(np.where(working_nodes > 0))  # 返回所有图节点的坐标
    valid_nodes = np.transpose(valid_nodes)  # valid_nodes 等价于 node_pixel

    for node_idx in range(num_nodes):
        y, x = node_pixel[node_idx]  # getting the position for current node
        labels.append(ref[y, x])  # CNN prediction提供的label信息
        #  基础的4领域连接
        for axis in range(2):
            axisy = int(axis == 0)
            axisx = int(axis == 1)
            for ne in [-1, 1]:
                neighbor = y + axisy * ne, x + axisx * ne
                if neighbor not in pixel_node:  # 如果四领域中的点不在ROI区域中,放弃该节点
                    continue
                ne_idx = pixel_node[neighbor]  # 否则拿出节点的ID
                if (node_idx, ne_idx) not in tabu_list and (ne_idx, node_idx) not in tabu_list:  # 判断某两个节点的边是否存在
                    tabu_list[(node_idx, ne_idx)] = 1  # adding the edge to the tabu list
                    # 给新增节点到中心节点的边上增加权重
                    weighting.weights_for((y, x), neighbor, args)  # 计算权重并保存到权重矩阵中
                    weighting.weights_for(neighbor, (y, x), args)  # 保存其对称位置的权重
                    edges.append([node_idx, ne_idx])  #
                    edges.append([ne_idx, node_idx])

        #  K个随机节点
        for j in range(k_random):
            valid_neigh = False
            if node_idx not in nodes_complete:
                nodes_complete[node_idx] = 0
            elif nodes_complete[node_idx] == k_random:
                break

            while not valid_neigh:
                lu_idx = np.random.randint(low=0, high=num_nodes)  # we look for a random node.
                yl, xl = valid_nodes[lu_idx]  # getting the euclidean coordinates for the voxel.
                lu_idx = pixel_node[yl, xl]  # getting the node index.
                if lu_idx not in nodes_complete:
                    nodes_complete[lu_idx] = 0
                    valid_neigh = True
                elif nodes_complete[lu_idx] < k_random:
                    valid_neigh = True

            if not (node_idx, lu_idx) in tabu_list and not (lu_idx, node_idx) in tabu_list \
                    and node_idx != lu_idx:  # checking if the edge was already generated
                weighting.weights_for((y, x), (yl, xl), args)  # computing the weight for the current pair.
                weighting.weights_for((yl, xl), (y, x), args)
                tabu_list[(node_idx, lu_idx)] = 1
                edges.append([node_idx, lu_idx])
                #  Adding the weight in the opposite direction
                edges.append([lu_idx, node_idx])
                #  Increasing the amount of neighbors connected to each node
                nodes_complete[node_idx] += 1
                nodes_complete[lu_idx] += 1
    edges = np.asarray(edges, dtype=int)  # [[0,40], [40,0]...........]    0-th到40-th节点上有边  40-th到0-th节点上有边
    pp_args = {
        "edges": edges,  # 所有的边   [266896, 2]
        "num_nodes": num_nodes  # 节点的个数  13373 * (4+16)
    }
    # print(weighting.weights1)   # 所有边上的期望相似性   例如0-th到40-th节点边上的权重
    weighting.post_process(pp_args)  # Applying weight post-processing, e.g. normalization
    weights = weighting.get_weights()  # weight1, weight2, weight3     # (13373, 13373)
    edges, weights, _ = sparse_to_tuple(weights)  # edges: (266896, 2)  weights: (266896,)
    return edges, weights, np.asarray(labels, dtype=np.float32), num_nodes


def graph_fts(fts, node_pixel):
    N = node_pixel.shape[0]  # 13373
    K = fts.shape[0]  # 每个节点的特征个数
    ft_mat = np.zeros(shape=(N, K), dtype=np.float32)  # 节点特征矩阵 13373行 5维
    for node_idx in range(N):
        h, w = node_pixel[node_idx]
        ft_mat[node_idx, :] = fts[:, h, w]
    return ft_mat


# Getting uncertain elements
def generate_mask(unc_image, node_pixel, th=0):
    num_nodes = node_pixel.shape[0]
    mask = np.zeros(shape=(num_nodes, 1), dtype=np.float32)
    for node_idx in range(num_nodes):
        y, x = node_pixel[node_idx]
        mask[node_idx] = float(unc_image[y, x] > th)
    return mask


def reference_to_graph(GT, node_pixel):
    N = node_pixel.shape[0]  # 13373个节点
    labels = np.zeros(shape=(N, 1), dtype=np.float32)  # [13373, 1]
    for node_idx in range(N):
        y, x = node_pixel[node_idx]
        labels[node_idx] = GT[y, x]
    return labels


def creating_graph():
    shadow_path = cashfile + "roi_shadow.npy"
    mask_path = cashfile + "roi_mask.npy"
    seg_path = cashfile + "roi_prediction.npy"
    var_path = cashfile + "bin_ent.npy"
    valid_path = cashfile + "graph_roi.npy"

    probability = np.load(cashfile + "roi_expectation.npy")  # ROI区域的平均预测
    entropy = np.load(cashfile + "roi_entropy.npy")  # ROI区域的交叉熵值
    seg_image = np.load(seg_path)  # 不要Dropout的前提下，ROI区域的标准预测
    var_image = np.load(var_path)  # ROI区域中高不确定性的像素位置
    GT = np.load(mask_path)  # 输入图像的GT
    shadow = np.load(shadow_path)  # 输入图像

    # -----ROI区域的阴影图像去均值除方差--------标准化
    num_pixel = float(shadow.shape[0] * shadow.shape[1] * shadow.shape[2])
    mean = shadow.astype(np.float32).sum() / num_pixel
    fangcha = np.sum((shadow.astype(np.float32) - mean) ** 2) / num_pixel
    fts = np.array(shadow, dtype=np.float32)
    fts = (fts - mean) / fangcha

    # 叠加ROI图像的3通道,期望值的1通道,熵值的1通道---->>>>5通道
    new_probability = np.expand_dims(probability, axis=0)
    new_entropy = np.expand_dims(entropy, axis=0)
    fts = np.concatenate((fts, new_probability, new_entropy), axis=0)  # 5, 144, 125

    # 加载Graph_ROI中需要重新验证的节点
    valid_nodes = np.load(valid_path)  # 144, 125
    # 生成一个字典，一个二维数组
    pixel_node, node_pixel = map_pixel_nodes(shadow.shape, valid_nodes.astype(np.bool))

    # 将特征转化为图representation  fts: [5, 144, 125]   node_pixel:[13373, 2]
    ft_graph = graph_fts(fts, node_pixel)  # [13373, 5]

    args = {
        "shadow": (shadow.astype(np.float32) - mean) / fangcha,  # 3, 144, 125
        "prediction": seg_image,  # 144, 125
        "probability": probability,  # 144, 125
        "uncertainty": var_image,  # 144, 125
        "entropy_map": entropy,  # 144, 125
        "features": fts  # 5, 144, 125
    }

    graph, weights, lb, N = connect_n6_krandom(ref=seg_image, pixel_node=pixel_node,
                                               node_pixel=node_pixel, working_nodes=valid_nodes,
                                               k_random=16, weighting=Weighting(),
                                               args=args)

    # graph:    哪两个ID的节点之间有边                            (266894, 2)   用来构建度矩阵，邻接矩阵
    # weights:  每条边上的权重：0.5 * w1 + w2 + w3               (266894,)     用于反向传播时提供相似性度量
    # lb:       CNN prediction without drop时提供的label信息    (13373,)      用于给ROI区域中的可靠像素提供标签1或者0,可靠的阴影像素，可靠的非阴影像素
    # N:        节点的个数                                      13373         等价于我们要生成图的数量

    # var_image  (144, 125)  limit区域   1标识高不确定点  0标识可靠点
    # node_pixel  (13373, 2) 存放ROI区域的坐标

    # 生成不确定性区域的mask

    uncertain_mask = generate_mask(var_image, node_pixel)  # mask的大小为 [13373, 1]  不确定的点有767个
    # Graph中节点在GT中的label信息       得到信息：13355个节点中在GT中有1387非阴影节点,11968阴影节点    761个节点是不确定性节点
    ref_label = reference_to_graph(GT, node_pixel)  # mask为GT  node_pixel  (13373, 2) 存放ROI区域的坐标
    # 但是在很有领域GT提供的label信息并不正确，这部分的值不可以用于训练

    np.save(cashfile + "graph.npy", graph)  # 哪两个ID的节点之间有边 (266894, 2)
    np.save(cashfile + "graph_weights.npy", weights)  # 每条边上的权重：0.5 * w1 + w2 + w3  (266894,)
    np.save(cashfile + "graph_node_features.npy", ft_graph)  # [13373, 5]  所有节点构成的特征矩阵
    np.save(cashfile + "graph_ground_truth.npy", lb)  # CNN prediction without drop时提供的label信息    (13373,)
    np.save(cashfile + "reference_graph.npy", ref_label)  # Graph中节点在GT中的label信息  这个GT参考价值不大
    np.save(cashfile + "unc_mask.npy", uncertain_mask)  # 不确定性的点，不参与训练的部分。以可靠像素样本为中心的构建的Graph才参与训练
    #
    print("------------构建图的结果-----------")
    print("图的形状: {}".format(graph.shape))
    print("权重的形状: {}".format(weights.shape))
    print("图特征的形状: {}".format(ft_graph.shape))  # RGB3通道,期望值的1通道,熵值的1通道
    print("训练样本的标签形状: {}".format(lb.shape))
    print("GT的标签形状: {}".format(ref_label.shape))
    print("不确定性像素mask的形状: {}".format(uncertain_mask.shape))
    print("参与计算的node数量: {}".format(np.sum(valid_nodes)))
    print("不可靠node的数量: {}".format(int(np.sum(uncertain_mask))))
    print("可靠的node的数量: {}".format(int(N - np.sum(uncertain_mask))))
    print("可靠的阴影node数量: {}".format(np.sum(lb[np.where(uncertain_mask == 0)[0]] == 1)))
    print("可靠的非阴影node数量: {}".format(np.sum(lb[np.where(uncertain_mask == 0)[0]] == 0)))
    #
    # info = {
    #     "N": N,  # 节点的数量
    #     "total_edges": N,  # 边的数量？？？  一个节点理论上有20条边！
    #     "graph_shape": graph.shape,
    #     "weight_shape": weights.shape,
    #     "ft_shape": ft_graph.shape,
    #     "train_labels_shape": lb.shape,
    #     "ref_labels_shape": ref_label.shape,
    #     "mask_uncertainty_shape": uncertain_mask.shape,
    #     "num_nodes": np.sum(valid_nodes),
    #     "num_uncertainty_nodes": np.sum(uncertain_mask),
    #     "num_certainty_nodes": N - np.sum(uncertain_mask),
    #     "num_positive_samples": np.sum(lb[np.where(uncertain_mask == 0)[0]] == 1),
    #     "num_negative_samples": np.sum(lb[np.where(uncertain_mask == 0)[0]] == 0)
    # }
if __name__ == '__main__':
    creating_graph()
