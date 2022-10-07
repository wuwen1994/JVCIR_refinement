import os
import cv2
import torch
from config import pre_path, label_path

def BER(pre_path, label_path):
    img_list = os.listdir(pre_path)
    sum_tp = 0.0
    sum_tn = 0.0
    sum_fp = 0.0
    sum_fn = 0.0
    for i, name in enumerate(img_list):
        if name.endswith('.png'):
            predict = cv2.imread(os.path.join(pre_path, name), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
            y_actual = torch.from_numpy(label).float()
            y_hat = torch.from_numpy(predict).float()
            y_hat = y_hat.ge(128).float()
            y_actual = y_actual.ge(128).float()
            y_actual = y_actual.squeeze(1)
            y_hat = y_hat.squeeze(1)
            pred_p = y_hat.eq(1).float()
            pred_n = y_hat.eq(0).float()
            pre_positive = float(pred_p.sum())
            pre_negtive = float(pred_n.sum())
            # FN
            fn_mat = torch.gt(y_actual, pred_p)
            FN = float(fn_mat.sum())
            # FP
            fp_mat = torch.gt(pred_p, y_actual)
            FP = float(fp_mat.sum())
            TP = pre_positive - FP
            TN = pre_negtive - FN
            sum_tp = sum_tp + TP
            sum_tn = sum_tn + TN
            sum_fp = sum_fp + FP
            sum_fn = sum_fn + FN
    pos = sum_tp + sum_fn
    neg = sum_tn + sum_fp

    if (pos != 0 and neg != 0):
        BAC = (.5 * ((sum_tp / pos) + (sum_tn / neg)))
    elif (neg == 0):
        BAC = (.5 * (sum_tp / pos))
    elif (pos == 0):
        BAC = (.5 * (sum_tn / neg))
    else:
        BAC = .5
    accuracy = (sum_tp + sum_tn) / (pos + neg) * 100
    global_ber = (1 - BAC) * 100
    return global_ber, accuracy


if __name__ == "__main__":
    ber, acc = BER(pre_path, label_path)
    print("BER:%.2f, Acc:%.2f" % (ber, acc))
