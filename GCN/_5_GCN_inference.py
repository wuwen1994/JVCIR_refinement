import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from GCN.config import shadow_dir, cashfile, sp
from utils import reconstruct_from_n4, map_pixel_nodes, npy_to_nifti, computer_ber


def GCN_inference(img, gt, model, adj, features, get_probs, gcn_th, i):
    roi_limits = np.load(cashfile + "limits.npy")  # roi的边界:矩形
    segmentation = np.load(cashfile + "roi_prediction.npy")  # roi区域的标准预测值
    gt[gt >= 0.3] = 1  # 将GT做阈值化
    gt[gt < 0.3] = 0

    roi_shadow = np.load(cashfile + "roi_shadow.npy")  # ROI区域的阴影图像  矩形

    valid_nodes = np.load(cashfile + "graph_roi.npy")  # ROI中通过膨胀操作得到的区域，包含所有的图节点（图工作区域）
    model.eval()
    output = model(features, adj)

    pixel_node, node_pixel = map_pixel_nodes(roi_shadow.shape, valid_nodes.astype(np.bool))
    if get_probs:
        graph_predictions = output.cpu().detach().numpy().astype(np.float32)
        graph_predictions = reconstruct_from_n4(graph_predictions, node_pixel, roi_shadow.shape, dtype=np.float)
        gp_expanded = np.zeros(img.shape[1:], dtype=np.float)
        gp_expanded[roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]] = graph_predictions
        return gp_expanded
    else:
        graph_predictions = (output > gcn_th).cpu().numpy().astype(np.float32)

    graph_predictions = reconstruct_from_n4(graph_predictions, node_pixel,
                                            roi_shadow.shape)  # recovering the volume shape

    refined = graph_predictions

    # 恢复原图大小，方便测试
    segmentation_expanded = np.zeros(img.shape[1:], dtype=np.float)
    segmentation_expanded[roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]] = segmentation

    refined_expanded = np.zeros(img.shape[1:], dtype=np.float)
    refined_expanded[roi_limits[0]:roi_limits[1], roi_limits[2]:roi_limits[3]] = refined

    np.save(cashfile + "graph_prediction.npy", refined)
    npy_to_nifti(refined_expanded, cashfile + "gcn_prediction.nii.gz")

    # 计算BER
    CNN_ber, CNN_acc = computer_ber(gt, segmentation_expanded)
    GCN_ber, GCN_acc = computer_ber(gt, refined_expanded)

    # 可视化
    image_path = shadow_dir[i]
    file_name = image_path.split(sp)[-1]
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
    plt.title("GCN_refined, BER: {}, ACC: {}%".format(round(GCN_ber, 2), round(GCN_acc, 2)))
    plt.imshow(refined_expanded)
    plt.show()
    # plt.savefig("./result/" + file_name, dpi=300)
    plt.close()

    info = {
        "cnn_ber": CNN_ber,
        "cnn_acc": CNN_acc,
        "gcn_ber": GCN_ber,
        "gcn_acc": GCN_acc
    }

    return info
