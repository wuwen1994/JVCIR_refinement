import cv2 as cv
import numpy as np


def visual(image, Nor=True):
    if Nor:
        image = (image * 255).astype(np.uint8)
    img_ = cv.applyColorMap(image, cv.COLORMAP_JET)
    cv.imwrite("heatmap.jpg", img_)
    heatmap = cv.imread("heatmap.jpg")
    cv.imshow("demo", heatmap)
    cv.waitKey()


if __name__ == "__main__":
    ROI_Entropy = np.load("prediction/ROI_Entropy.npy")

    visual(ROI_Entropy)
