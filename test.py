import os
import cv2
import glob
import torch
import torchvision
import numpy as np
from evaluate import BER
import os.path as osp
from torch.utils import data
from dataset import Dataset_test
from model.ResNet import ResNet as MyModel
# from model.MobileNet import MoblieNet as MyModel
from config import test_images_dir, sp, pre_path, label_path, precess_dir, pth


def test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)  # 某一张图片的输出，5个通道  [5, batch, 1, 256, 256]  list类型
        all_results = torch.zeros((len(results), 1, H, W))  # 初始化一个全0的[5, 1, 256, 256]的tensor
        for i in range(len(results)):
            all_results[i, 0, :, :] = results[i]
        filename = osp.splitext(test_list[idx])[0].split(sp)[-1]
        # torchvision.utils.save_image(all_results, osp.join(save_dir, '%s.jpg' % filename))
        result = torch.sigmoid(results[-1])
        result[result >= 0.3] = 1
        result[result < 0.3] = 0
        fuse_res = torch.squeeze(result.detach()).cpu().numpy()
        fuse_res = (fuse_res * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s.png' % filename), fuse_res)


def precess(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)  # 某一张图片的输出，5个通道  [5, batch, 1, 256, 256]  list类型
        # for i, result in enumerate(results):
        #     filename = osp.splitext(test_list[idx])[0].split(sp)[-1] + "_" + str(i)
        #     result = torch.sigmoid(result)
        #     # result[result >= 0.3] = 1
        #     # result[result < 0.3] = 0
        #     fuse_res = torch.squeeze(result.detach()).cpu().numpy()
        #     fuse_res = (fuse_res * 255).astype(np.uint8)
        #     cv2.imwrite(osp.join(save_dir, '%s.png' % filename), fuse_res)
        #
        filename = osp.splitext(test_list[idx])[0].split(sp)[-1]
        result = torch.sigmoid(results[-1])
        result[result >= 0.3] = 1
        result[result < 0.3] = 0
        fuse_res = torch.squeeze(result.detach()).cpu().numpy()
        fuse_res = (fuse_res * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s.jpg' % filename), fuse_res)


if __name__ == "__main__":
    Batch_size = 1
    test_list = sorted(glob.glob(test_images_dir))
    sample_list = sorted(glob.glob(precess_dir))
    print("-----------Testing Start------------")
    test_ds = Dataset_test(test_list)
    test_dl = data.DataLoader(test_ds, batch_size=Batch_size, shuffle=False)

    sample_ds = Dataset_test(sample_list)
    sample_dl = data.DataLoader(sample_ds, batch_size=Batch_size, shuffle=False)


    if osp.isfile(pth):
        if torch.cuda.is_available():
            model = MyModel().cuda()
            model.load_state_dict(torch.load(pth))  # 使用GPU测试
            print("-----------Loading checkpoint to GPU-----------")

        else:
            model = MyModel()
            model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))  # 使用CPU测试
            print("-----------Loading checkpoint to CPU-----------")
    else:
        print("-----------No checkpoint-----------")

    print("-----------Single-scale Testing Start-----------")


    precess(model, test_dl, test_list, pre_path)
    # precess(model, sample_dl, sample_list, pre_path)
