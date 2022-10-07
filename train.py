import os
import torch
import glob
import time
from torch import nn
from tqdm import tqdm
from utils import loss_fn
from evaluate import BER
from torch.utils import data
from test import test
from model.ResNet import ResNet as MyModel
# from model.MobileNet import MoblieNet as MyModel
from dataset import Dataset_train, Dataset_test
from config import train_images_dir, train_annos_dir, test_images_dir, pre_path, label_path, \
    train_name, Batch_size, Deep_supervision
import datetime


def train(model, train_loader, epoch):
    train_loss = 0
    model.train()
    # 训练
    for x, y in tqdm(train_loader):  # i表示i-th batch
        x, y = x.cuda(), y.cuda()
        _y_pred = model(x)  # _y_pred接收的类型为list
        if not Deep_supervision:
            # 单监督
            y, y_pred = torch.squeeze(y), torch.squeeze(_y_pred[-1])
            loss = loss_fn(-1, y_pred, y)
        else:
            # 深监督
            loss = torch.zeros(1).cuda()
            for j, y_pred in enumerate(_y_pred):
                y, y_pred = torch.squeeze(y), torch.squeeze(y_pred)
                loss += loss_fn(j, y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
    train_loss = train_loss / len(train_loader.dataset)
    # 计算测试集BER
    with torch.no_grad():
        test(model, test_dl, test_images, pre_path)
        ber, acc = BER(pre_path, label_path)

    # -----------------------保存模型-----------------------------------
    PATH = basic_path_dir + train_name + '_' + str(epoch + 1) + '_' + str(ber)[:4] + '_' + str(acc)[:4] + '.pth'
    torch.save(model.state_dict(), PATH)

    spl = "	"  # excell 分隔符
    log = str(epoch + 1) + spl + str(round(train_loss, 3)) + spl + str(round(ber, 2)) + spl + str(
        round(acc, 2))
    print(log)
    dt = datetime.datetime.now().strftime('%Y-%m-%d')
    with open(train_name + "_" + dt + "_" + "log.txt", "a") as f:
        f.write(log + "\n")


if __name__ == "__main__":
    # 提取训练和测试数据的路径
    train_images = sorted(glob.glob(train_images_dir))
    train_annos = sorted(glob.glob(train_annos_dir))

    test_images = sorted(glob.glob(test_images_dir))

    # 创建DataSet实例,创建DataLoader实例
    train_ds = Dataset_train(train_images, train_annos)
    train_dl = data.DataLoader(train_ds, batch_size=Batch_size, shuffle=True)

    test_ds = Dataset_test(test_images)
    test_dl = data.DataLoader(test_ds, batch_size=1)
    # ---------------------------------------模型初始化--------------------------------------------
    model = MyModel()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    # -----------------------------------训练开始--------------------------
    basic_path_dir = './MyPth/'
    if not os.path.exists(basic_path_dir):
        os.makedirs(basic_path_dir)

    epochs = 30
    time_start = time.time()
    for epoch in range(epochs):
        train(model, train_dl, epoch)
    time_end = time.time()
    print('totally cost', time_end - time_start)
