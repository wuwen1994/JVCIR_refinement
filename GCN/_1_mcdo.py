import torch
import time
import numpy as np
from utils import npy_to_nifti
from GCN.config import cashfile


def MCDO(model, num_of_MCDO, shadow, mask):
    # 计算测试数据的标准输出,将所有的结果存放到seg_results
    torch.set_grad_enabled(False)  # 预测阶段 不计算梯度
    model.eval()
    shadow = shadow.cuda()
    predictions = model(shadow)
    _mask = torch.sigmoid(predictions[-1])
    result_ = torch.squeeze(_mask).cpu().numpy()  # 去掉维度为1的通道
    result_[result_ > 0.3] = 1
    result_[result_ <= 0.3] = 0
    result = result_.astype(np.uint8)  # 转化
    shadow_np = torch.squeeze(shadow.detach()).cpu().numpy()
    mask_np = torch.squeeze(mask.detach()).cpu().numpy()
    np.save(cashfile + "shadow.npy", shadow_np)
    np.save(cashfile + "mask.npy", mask_np)

    # 以numpy的形式的保存以上预测结果

    # 保存了历史预测结果
    cnn_prediction_dir = cashfile + "cnn_prediction.nii.gz" + ""
    npy_to_nifti(result, cnn_prediction_dir)

    def bounding_cube(vol):
        a = np.where(vol != 0)
        box = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return box

    # 找到roi的边界
    roi_limits = bounding_cube(result)

    # 保存ROI信息
    ROI_save_dir = cashfile + "limits.npy"
    np.save(ROI_save_dir, np.asanyarray(roi_limits))

    # roi区域的标准预测值,原始图像的roi区域,GT图像的roi区域
    roi_prediction = result[roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]]
    roi_shadow = shadow_np[:, roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]]
    roi_mask = mask_np[roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]]

    # 保存以上结果
    np.save(cashfile + "roi_prediction.npy", roi_prediction)
    np.save(cashfile + "roi_shadow.npy", roi_shadow)
    np.save(cashfile + "roi_mask.npy", roi_mask)

    # 再次读取测试数据
    # print("开始Montecarlo Dropout计算")
    time_start = time.time()
    # print("Running for {} samples".format(num_of_MCDO))

    mc_outs = []
    # 开始预测
    torch.set_grad_enabled(False)
    model.train()
    for t in range(num_of_MCDO):
        prediction = torch.squeeze(torch.sigmoid(model(shadow)[-1])).cpu().numpy()
        mc_outs.append(prediction)

    # Expectation 20轮的预测平均值
    Expectation = np.sum(mc_outs, axis=0, dtype=np.float) / float(num_of_MCDO)
    Entropy = -Expectation * np.log2(Expectation + 1.0e-15) - (1.0 - Expectation) * np.log2(1.0 - Expectation + 1.0e-15)
    # 保存Expectation, Entropy
    np.save(cashfile + "expectation.npy", Expectation)
    np.save(cashfile + "entropy.npy", Entropy)
    # 将熵值小于0的部分归0
    Entropy[Entropy < 0] = 0.0

    # 保存在期望和熵为NIFTI格式
    npy_to_nifti(Expectation * 255, cashfile + "cnn_expectation.nii.gz" + "")
    max_ent = np.max(Entropy)
    npy_to_nifti((Entropy * (255 // max_ent)).astype(np.uint8), cashfile + "cnn_entropy.nii.gz" + "")

    # 对期望阈值化
    trsh_Expectation = (Expectation > 0.5).astype(np.int)

    # 获取ROI区域的期望和熵
    Expectation_roi = Expectation[roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]]
    Entropy_roi = Entropy[roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]]

    # 保存ROI区域的期望和熵
    np.save(cashfile + "roi_expectation.npy", Expectation_roi)
    np.save(cashfile + "roi_entropy.npy", Entropy_roi)
    time_end = time.time()
    # print('totally cost', time_end - time_start)
    return trsh_Expectation.shape, shadow_np, mask_np
