import glob
import os
import os.path as osp

# GCN在哪个数据集上测试
Linux = False
dataset_name = "ISTD"

if Linux:
    file_path = "/home/featurize/work/Datasets/"
    sp = "/"
else:
    file_path = "F:/Datasets/"
    sp = "\\"

# 数据集位置
basic_dir = file_path + dataset_name + "/test"
pth = "../MyPth/" + dataset_name + ".pth"
cashfile = "./cashfile/"
save_dir = "./result/"
if not osp.isdir(cashfile):
    os.makedirs(cashfile)
if not osp.isdir(save_dir):
    os.makedirs(save_dir)
# 测试数据路径
shadow_dir = sorted(glob.glob(basic_dir + "/shadow/*.jpg"))
mask_dir = sorted(glob.glob(basic_dir + "/mask/*.png"))
