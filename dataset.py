import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

# ----------------数据增大和Tensor化------------------------

# 创建transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ----------------Dataset类------------------------
class Dataset_train(data.Dataset):
    def __init__(self, imgs_path, anno_path):
        self.imgs_path = imgs_path
        self.annos_path = anno_path

    def __getitem__(self, index):
        # 路径
        img = self.imgs_path[index]
        anno = self.annos_path[index]

        # 读取
        image = Image.open(img)
        mask = Image.open(anno)

        # 增强
        image = transform(image)
        mask = transform(mask)

        # 将mask阈值化
        mask[mask >= 0.3] = 1
        mask[mask < 0.3] = 0

        # 去掉为1的维度
        mask = torch.squeeze(mask)
        return image, mask

    def __len__(self):
        return len(self.imgs_path)


class Dataset_test(data.Dataset):
    def __init__(self, imgs_path):
        self.imgs_path = imgs_path

    def __getitem__(self, index):
        img = self.imgs_path[index]
        image = Image.open(img)
        image = transform(image)
        return image

    def __len__(self):
        return len(self.imgs_path)
