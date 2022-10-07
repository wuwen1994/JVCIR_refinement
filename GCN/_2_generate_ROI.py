import numpy as np
import scipy.ndimage as ndimage
from GCN._1_mcdo import npy_to_nifti
from GCN.config import cashfile
from GCN.visualization import visual

def generate_ROI(ref_shape):
    entropy_th = 0.80  # 在20次预测中，有百分之75的次数某个点被预测为阴影，称该点为确定性样本，否则为不确定性样本，后期进行优化
    # 加载ROI区域的entropy
    Entropy_roi = np.load(cashfile + "roi_entropy.npy")
    # 加载ROI区域的期望
    Expectation_roi = np.load(cashfile + "roi_expectation.npy")
    # 加载初始ROI边界
    roi = np.load(cashfile + "limits.npy")
    # print("Selecting Pixels for Graph ROI...")
    # print("Entropy shape: {}".format(Entropy_roi.shape))
    # print("Probability shape: {}".format(Expectation_roi.shape))
    # print("Entropy th is {}".format(entropy_th))
    # 阈值化ROI区域的Entropy
    bin_Entropy_roi = (Entropy_roi > entropy_th).astype(np.uint8)
    # visual(bin_Entropy_roi, True)
    # 阈值化ROI区域的期望
    bin_Expectation_roi = (Expectation_roi > 0.5).astype(np.uint8)
    # visual(bin_Expectation_roi, True)
    # 定义膨胀操作的kernel
    kernel = np.ones(shape=(7, 7), dtype=np.bool)
    # 对bin_Entropy_roi膨胀操作
    dilated = ndimage.binary_dilation(bin_Entropy_roi, structure=kernel).astype(np.uint8)
    # visual(dilated, True)
    # print("Input nodes: {} reduced to {}".format(Expectation_roi.shape[0] * Expectation_roi.shape[1], np.sum(dilated)))
    # 与阈值化后的期望ROI做并集
    dilated = ((dilated + bin_Expectation_roi) > 0).astype(np.int)
    # visual(dilated, True)

    # 扩大二值交叉熵至全局层面
    expanded_bin_entropy = np.zeros(shape=ref_shape)
    expanded_bin_entropy[roi[0]:roi[1], roi[2]:roi[3]] = bin_Entropy_roi
    # visual(expanded_bin_entropy, True)

    # 保存要构建Graph的范围，正方形。
    np.save(cashfile + "graph_roi.npy", dilated)
    # visual(dilated, True)
    # 保存ROI区域中高不确定性的像素位置，其中不确定性高的像素，为GCN标签传播的时候优化的对象
    np.save(cashfile + "bin_ent.npy", bin_Entropy_roi)
    # 保存全局范围内，不可靠像素点的位置
    npy_to_nifti(expanded_bin_entropy, cashfile + "binary_cnn_entropy.nii.gz" + "")
    # print("ROI for Graph Done!")
