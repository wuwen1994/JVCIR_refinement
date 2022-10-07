import torch
from torchvision import models
from torch import nn
from torch.autograd import Variable
from torchsummary import summary

# ------------------------------创建模型----------------------------------
pretrain_model = models.mobilenet_v2(pretrained=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class MoblieNet(nn.Module):
    def __init__(self, n_of_c=8):
        super(MoblieNet, self).__init__()
        # 从moblie-v2中剥离出5个卷积块                 256*256*3
        self.Block1 = pretrain_model.features[:2]  # 128*128*16    stage1
        self.Block2 = pretrain_model.features[2:4]  # 64*64*24     stage2
        self.Block3 = pretrain_model.features[4:7]  # 32*32*32     stage3
        self.Block4 = pretrain_model.features[7:14]  # 16*16*96    stage4
        self.Block5 = pretrain_model.features[14:]  # 8*8*1280     stage5

        # 2倍上采样
        self.upsample_2x = nn.ConvTranspose2d(n_of_c, n_of_c, 4, 2, 1, bias=False)
        # 4倍上采样
        self.upsample_4x = nn.ConvTranspose2d(n_of_c, n_of_c, 8, 4, 2, bias=False)
        # 8倍上采样
        self.upsample_8x = nn.ConvTranspose2d(n_of_c, n_of_c, 16, 8, 4, bias=False)
        # 16倍上采样
        self.upsample_16x = nn.ConvTranspose2d(n_of_c, n_of_c, 32, 16, 8, bias=False)
        # 32倍上采样
        self.upsample_32x = nn.ConvTranspose2d(n_of_c, n_of_c, 64, 32, 16, bias=False)

        self.channl_attention1 = ChannelAttention(16)
        self.channl_attention2 = ChannelAttention(24)
        self.channl_attention3 = ChannelAttention(32)
        self.channl_attention4 = ChannelAttention(96)
        self.channl_attention5 = ChannelAttention(1280)

        self.Compression1 = nn.Conv2d(16, n_of_c, 1)
        self.Compression2 = nn.Conv2d(24, n_of_c, 1)
        self.Compression3 = nn.Conv2d(32, n_of_c, 1)
        self.Compression4 = nn.Conv2d(96, n_of_c, 1)
        self.Compression5 = nn.Conv2d(1280, n_of_c, 1)

        self.Compress = nn.Conv2d(n_of_c, 1, 1)
        self.fusion = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        stage1 = self.Block1(x)  # 128*128  16
        s1 = self.Compression1(stage1 * self.channl_attention1(stage1))  # 128*128 8
        s1 = self.upsample_2x(s1)  # 256*256 8
        s1 = self.Compress(s1)  # 256*256 1

        stage2 = self.Block2(stage1)  # 64*64    24
        s2 = self.Compression2(stage2 * self.channl_attention2(stage2))  # 64*64 8
        s2 = self.upsample_4x(s2)  # 256*256 8
        s2 = self.Compress(s2)  # 256*256 1

        stage3 = self.Block3(stage2)
        s3 = self.Compression3(stage3 * self.channl_attention3(stage3))
        s3 = self.upsample_8x(s3)
        s3 = self.Compress(s3)

        stage4 = self.Block4(stage3)
        s4 = self.Compression4(stage4 * self.channl_attention4(stage4))
        s4 = self.upsample_16x(s4)
        s4 = self.Compress(s4)

        stage5 = self.Block5(stage4)
        s5 = self.Compression5(stage5 * self.channl_attention5(stage5))
        s5 = self.upsample_32x(s5)
        s5 = self.Compress(s5)

        fuse = torch.cat((s1, s2, s3, s4, s5), dim=1)  # 256*256*5
        s6 = self.fusion(fuse)

        feature_maps = [s1, s2, s3, s4, s5, s6]
        return feature_maps


if __name__ == "__main__":
    print('-----' * 5)
    rgb = torch.randn(1, 3, 256, 256)
    rgb = Variable(rgb).cuda()
    model = MoblieNet()
    model.cuda()
    out = model(rgb)
    print(summary(model, input_size=(3, 256, 256), batch_size=-1))
    print(out.shape)
    # from thop import profile
    # flops, params = profile(model, inputs=(rgb,))
    # print("flops为"+str(flops/1000000000))
    # print("params为"+str(params/1000000))
