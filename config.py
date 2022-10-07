import pynvml  # pip install nvidia-ml-py

Linux = False
dataset_name = "ISTD"
Deep_supervision = True
# ---------------指定数据路径(无需改动)--------------------
if Linux:
    basic_dir = "/home/featurize/work/Datasets/"
    sp = "/"
else:
    basic_dir = "F:/Datasets/"
    sp = "\\"

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
if meminfo.total / (2 ** 30) >= 20:
    Batch_size = 48
elif meminfo.total / (2 ** 30) >= 10:
    Batch_size = 20
else:
    Batch_size = 18
print("显存:%s GB" % str(meminfo.total / (2 ** 30)))
print("batch_size:", Batch_size)
basic_dir = basic_dir + dataset_name
train_images_dir = basic_dir + "/train/shadow/*.jpg"
train_annos_dir = basic_dir + "/train/mask/*.png"

test_images_dir = basic_dir + "/test/shadow/*.jpg"
precess_dir = "./sample/*.*"
pre_path = './result/'
label_path = basic_dir + "/test/mask_resize"
pth = "./MyPth/" + dataset_name + ".pth"
