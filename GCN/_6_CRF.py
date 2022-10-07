import numpy as np
import torch
from PIL import Image
import pydensecrf.densecrf as dcrf
from model.ResNet import ResNet as MyModel
from GCN._1_mcdo import MCDO
from utils import create_pairwise_gaussian, create_pairwise_bilateral, computer_ber, npy_to_nifti
import matplotlib.pyplot as plt
from torch.utils import data
from dataset import Dataset_train as Dataset
from GCN.config import shadow_dir, mask_dir, pth, cashfile, sp


def crf(shadow, gt, it, i):
    roi_shadow = np.load(cashfile + "roi_shadow.npy")  # ROI区域的阴影图像
    segmentation = np.load(cashfile + "roi_prediction.npy")  # CNN Prediction
    gt[gt >= 0.3] = 1  # 将GT做阈值化
    gt[gt < 0.3] = 0

    roi_limits = np.load(cashfile + "limits.npy")
    cnn_prediction = np.load(cashfile + "roi_prediction.npy")

    _, y, x = roi_shadow.shape

    U = np.ndarray(shape=[2, y, x], dtype=np.float32)
    print(cnn_prediction)
    U[0, :] = 1 - cnn_prediction
    U[1, :] = cnn_prediction

    d = dcrf.DenseCRF(x * y, 2)  # npoints, nlabels

    U = U.reshape((2, -1))
    d.setUnaryEnergy(-np.log(U + 1.0e-15))
    # y = 176  x = 152

    # 空间势能
    pairwise_gauss = create_pairwise_gaussian(sdims=(3, 3), shape=roi_shadow.shape[1:])
    d.addPairwiseEnergy(pairwise_gauss, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # 颜色势能
    pairwise_bilat = create_pairwise_bilateral(sdims=(80, 80), schan=(1,), img=roi_shadow, chdim=0)
    d.addPairwiseEnergy(pairwise_bilat, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(it)
    refined = np.argmax(Q, axis=0).reshape((y, x))

    # recovering sizes
    segmentation_expanded = np.zeros(shadow.shape[1:], dtype=np.float)
    segmentation_expanded[roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]] = segmentation

    refined_expanded = np.zeros(shadow.shape[1:], dtype=np.float)
    refined_expanded[roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]] = refined

    # 计算BER
    CNN_ber, CNN_acc = computer_ber(gt, segmentation_expanded)
    CRF_ber, CRF_acc = computer_ber(gt, refined_expanded)

    # 可视化
    image_path = shadow_dir[i]
    file_name = image_path.split("\\")[-1]
    image = Image.open(image_path)
    image = image.resize((256, 256))
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title('Image')
    plt.imshow(image)
    plt.subplot(2, 2, 2)
    plt.title('GT')
    plt.imshow(gt)
    plt.subplot(2, 2, 3)
    plt.title("CNN_predicted, BER: {}, ACC: {}%".format(round(CNN_ber, 2), round(CNN_acc, 2)))
    plt.imshow(segmentation_expanded)
    plt.subplot(2, 2, 4)
    plt.title("CRF_refined, BER: {}, ACC: {}%".format(round(CRF_ber, 2), round(CRF_acc, 2)))
    plt.imshow(refined_expanded)
    plt.savefig("./CRF_refined_result/" + file_name, dpi=300)
    plt.show()
    plt.close()

    np.save(cashfile + "crf_prediction.npy", refined)
    npy_to_nifti(refined_expanded, cashfile + "crf_prediction.nii.gz")

    info = {
        "cnn_ber": CNN_ber,
        "cnn_acc": CNN_acc,
        "gcn_ber": CRF_ber,
        "gcn_acc": CRF_acc
    }
    return info


if __name__ == '__main__':
    # 测试数据路径
    testing_dataset = Dataset(shadow_dir, mask_dir)
    testing_dl = data.DataLoader(testing_dataset, batch_size=1)

    for i, (shadow, mask) in enumerate(testing_dl):
        model = MyModel().cuda()
        model.load_state_dict(torch.load(pth))
        ref_shape, x, y = MCDO(model, 20, shadow, mask)
        crf_info = crf(x, y, 50, i)
