import torch.nn.functional as F
import torch
from GCN.BER import BER
import math
import numpy as np
import scipy.sparse as sp
import nibabel as nb
from numbers import Number

def loss_fn(i, prediction, label):
    lamda = [1., 1., 1., 1., 1., 1.]
    cost = F.binary_cross_entropy_with_logits(prediction, label)
    return cost * lamda[i]


def supervised_loss(prediction, label):
    mask = label.clone()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = F.binary_cross_entropy(prediction, label, weight=mask)
    return cost  # torch.cuda.FloatTensor


def entropy_loss(*prediction):
    predictions = prediction[0]  # torch.Tensor
    result = 0
    for p in predictions:
        p = p.item() + 1.0e-15  # 避免概率为0和为1是报错
        result += -  (p * math.log(p, 2) + abs(1 - p) * math.log(abs(1 - p), 2))
    mean_entopy = result / (len(predictions))
    return mean_entopy


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """行归一化"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """行归一化"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = (output > 0.5).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def reconstruct_from_n4(ft_mat, map_vector, shape, dtype=np.uint8):
    _, ys, xs  = shape
    N = map_vector.shape[0]
    rec_vol = np.zeros(shape=(ys, xs), dtype=dtype)
    for i in range(N):
        y, x  = map_vector[i]
        rec_vol[y, x] = dtype(ft_mat[i])
    return rec_vol

def map_pixel_nodes(shape, include_nodes):
    """
    给Graph中ROI区域的像素分类一个节点ID
    :param shape: 阴影图像的shape
    :param include_nodes:一个二值图像，为1的为ROI区域
    :return: 一个像素指向节点ID的字典，  一个从节点ID到像素坐标位置的数组
    """
    channel, height, width = shape
    N = np.sum(include_nodes.astype(np.int))  # ROI区域覆盖的所有像素个数
    pixel_node = {}
    node_pixel = np.zeros(shape=(N, 2), dtype=np.int)
    node_index = 0
    for h in range(height):
        for w in range(width):
            if not include_nodes[h, w]:  # 如果该像素为ROI区域的话
                continue
            node_pixel[node_index] = [h, w]  # 把ROI区域的像素，依次从1开始排序，然后把其坐标存到node_pixel中去
            pixel_node[h, w] = node_index
            node_index += 1
    return pixel_node, node_pixel


def npy_to_nifti(npy_vol, filename):
    byte_vol = np.round(npy_vol).astype(np.uint8)
    nifti = nb.Nifti1Image(byte_vol, None)
    nb.save(nifti, filename)

def create_pairwise_gaussian(sdims, shape):
    """
    Util function that create pairwise gaussian potentials. This works for all
    image dimensions. For the 2D case does the same as
    `DenseCRF2D.addPairwiseGaussian`.

    Parameters
    ----------
    sdims: list or tuple
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseGaussian`.
    shape: list or tuple
        The shape the CRF has.

    """
    # create mesh
    hcord_range = [range(s) for s in shape]
    mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s
    return mesh.reshape([len(sdims), -1])


def create_pairwise_bilateral(sdims, schan, img, chdim=-1):
    """
    Util function that create pairwise bilateral potentials. This works for
    all image dimensions. For the 2D case does the same as
    `DenseCRF2D.addPairwiseBilateral`.

    Parameters
    ----------
    sdims: list or tuple
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseBilateral`.
    schan: list or tuple
        The scaling factors per channel in the image. This is referred to
        `srgb` in `DenseCRF2D.addPairwiseBilateral`.
    img: numpy.array
        The input image.
    chdim: int, optional
        This specifies where the channel dimension is in the image. For
        example `chdim=2` for a RGB image of size (240, 300, 3). If the
        image has no channel dimension (e.g. it has only one channel) use
        `chdim=-1`.

    """
    # Put channel dim in right position
    if chdim == -1:
        # We don't have a channel, add a new axis
        im_feat = img[np.newaxis].astype(np.float32)
    else:
        # Put the channel dim as axis 0, all others stay relatively the same
        im_feat = np.rollaxis(img, chdim).astype(np.float32)

    # scale image features per channel
    # Allow for a single number in `schan` to broadcast across all channels:
    if isinstance(schan, Number):
        im_feat /= schan
    else:
        for i, s in enumerate(schan):
            im_feat[i] /= s

    # create a mesh
    cord_range = [range(s) for s in im_feat.shape[1:]]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s

    feats = np.concatenate([mesh, im_feat])
    return feats.reshape([feats.shape[0], -1])


def computer_ber(gt, prediction):
    gt = torch.from_numpy(gt).float() * 255
    prediction = torch.from_numpy(prediction).float() * 255
    return BER(gt, prediction)
