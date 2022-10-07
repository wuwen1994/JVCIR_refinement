from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchsummary import summary
import numpy as np

out_channels = [64, 128, 256, 512, 1024]


# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_c):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_c, in_c // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_c // 16, in_c, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.Dropout(p=0.3),  # MCDO
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.Dropout(p=0.3)  # MCDO
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, n_of_c=24):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)
        xuxian_shortcut1 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.conv2_1 = ResidualBlock(64, 128, 1, xuxian_shortcut1)
        self.conv2_2 = ResidualBlock(128, 128, 1)

        xuxian_shortcut2 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.conv3_1 = ResidualBlock(128, 256, 2, xuxian_shortcut2)
        self.conv3_2 = ResidualBlock(256, 256, 1)

        xuxian_shortcut3 = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512)
        )
        self.conv4_1 = ResidualBlock(256, 512, 2, xuxian_shortcut3)
        self.conv4_2 = ResidualBlock(512, 512, 1)

        xuxian_shortcut4 = nn.Sequential(
            nn.Conv2d(512, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.conv5_1 = ResidualBlock(512, 1024, 1, xuxian_shortcut4)
        self.conv5_2 = ResidualBlock(1024, 1024, 1)

        self.channl_attention1 = ChannelAttention(out_channels[0])
        self.channl_attention2 = ChannelAttention(out_channels[1])
        self.channl_attention3 = ChannelAttention(out_channels[2])
        self.channl_attention4 = ChannelAttention(out_channels[3])
        self.channl_attention5 = ChannelAttention(out_channels[4])

        # 初步压缩
        self.Compression1 = nn.Conv2d(out_channels[0], n_of_c, 1)
        self.Compression2 = nn.Conv2d(out_channels[1], n_of_c, 1)
        self.Compression3 = nn.Conv2d(out_channels[2], n_of_c, 1)
        self.Compression4 = nn.Conv2d(out_channels[3], n_of_c, 1)
        self.Compression5 = nn.Conv2d(out_channels[4], n_of_c, 1)

        # 2倍上采样
        self.upsample_2x = nn.ConvTranspose2d(n_of_c, n_of_c, 4, 2, 1, bias=False)
        # 4倍上采样
        self.upsample_4x = nn.ConvTranspose2d(n_of_c, n_of_c, 8, 4, 2, bias=False)
        # 8倍上采样
        self.upsample_8x = nn.ConvTranspose2d(n_of_c, n_of_c, 16, 8, 4, bias=False)
        # 16倍上采样
        self.upsample_16x = nn.ConvTranspose2d(n_of_c, n_of_c, 32, 16, 8, bias=False)

        # 压缩
        self.Compression = nn.Conv2d(n_of_c, 1, 1)

        # fusion
        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        stage1 = self.conv1(x)  # 64, 128, 128
        maxpool = self.maxpool1(stage1)  # 64, 64, 64
        # 2
        conv2_1 = self.conv2_1(maxpool)  # 128, 64, 64
        stage2 = self.conv2_2(conv2_1)  # 128, 64, 64
        # 2
        conv3_1 = self.conv3_1(stage2)  # 256, 32, 32
        stage3 = self.conv3_2(conv3_1)  # 256, 32, 32
        # 2
        conv4_1 = self.conv4_1(stage3)  # 512, 16, 16
        stage4 = self.conv4_2(conv4_1)  # 512, 16, 16
        # 2
        conv5_1 = self.conv5_1(stage4)  # 1024, 16, 16
        stage5 = self.conv5_2(conv5_1)  # 1024, 16, 16

        # 先注意力再压缩到24通道
        stage1 = self.Compression1(stage1 * self.channl_attention1(stage1))
        stage2 = self.Compression2(stage2 * self.channl_attention2(stage2))
        stage3 = self.Compression3(stage3 * self.channl_attention3(stage3))
        stage4 = self.Compression4(stage4 * self.channl_attention4(stage4))
        stage5 = self.Compression5(stage5 * self.channl_attention5(stage5))
        # 上采样到原尺度
        stage1 = self.upsample_2x(stage1)
        stage2 = self.upsample_4x(stage2)
        stage3 = self.upsample_8x(stage3)
        stage4 = self.upsample_16x(stage4)
        stage5 = self.upsample_16x(stage5)
        # 再压缩到1通道
        stage1 = self.Compression(stage1)
        stage2 = self.Compression(stage2)
        stage3 = self.Compression(stage3)
        stage4 = self.Compression(stage4)
        stage5 = self.Compression(stage5)
        # fusion成5通道，再压缩成单通道
        fuse = torch.cat((stage1, stage2, stage3, stage4, stage5), dim=1)
        result = self.fuse(fuse)
        # 输出中间结果
        feature_maps = [stage1, stage2, stage3, stage4, stage5, result]
        return feature_maps


if __name__ == "__main__":
    print('-----' * 5)
    rgb = torch.randn(1, 3, 256, 256)
    rgb = Variable(rgb).cuda()
    model = ResNet()
    model.cuda()
    out = model(rgb)
    print(summary(model, input_size=(3, 256, 256), batch_size=-1))
    print(out[-1].shape)
