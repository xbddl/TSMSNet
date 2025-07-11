# -*- coding: utf-8 -*-
"""
@Time    : 2023/1/11/011 12:35
@Author  : NDWX
@File    : network.py
@Software: PyCharm
"""
import torch
from torch import nn
from modules.new2_multi_sa.Multi_Modulation import Multi_Modulation,Multi_Modulation_Block
from modules.new2_multi_sa.improve_PAPPM import PAPPM
from modules.new2_multi_sa.RAF_Pag import RelationAwareFusion
# from Multi_Modulation import Multi_Modulation,Multi_Modulation_Block
# from improve_PAPPM import PAPPM
# from RAF_Pag import RelationAwareFusion
import pdb
from thop import profile

# from ptflops import get_model_complexity_info

__all__ = ['UNet_EF']


# 服务器用
def osjv(x1, x2):
    # x1 = torch.randn(4, 3, 256, 256)
    # x2 = torch.randn(4, 3, 256, 256)
    # x11 = x1.transpose(1, 3)
    # x22 = x2.transpose(1, 3)
    euclidean_distance = torch.pairwise_distance(x1, x2)
    euclidean_distance1 =  torch.unsqueeze(euclidean_distance, dim=1)
    return euclidean_distance1

# 电脑端用
# def osjv(x1, x2):
#     # x1 = torch.randn(4, 3, 256, 256)
#     # x2 = torch.randn(4, 3, 256, 256)
#     x11 = x1.transpose(1, 3)
#     x22 = x2.transpose(1, 3)
#     euclidean_distance = torch.pairwise_distance(x11, x22)
#     euclidean_distance1 = euclidean_distance.transpose(1, 2)
#     euclidean_distance2 = torch.unsqueeze(euclidean_distance1, dim=1)
#     return euclidean_distance2


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, status=True):
        super(ConvBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = Multi_Modulation_Block(mid_ch,status=status)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.status = status

    def forward(self, x):
        x = self.conv1(x)
        x_bn1 = self.bn1(x)
        x_act = self.activation(x_bn1)
        x_conv2,pred_map = self.conv2(x_act)
        # x_conv2 = self.bn2(x_conv2)
        output = x_conv2
        return output,pred_map


class ConvIDCBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, status=True):
        super(ConvIDCBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = Multi_Modulation_Block(mid_ch,status=status)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.status = status

    def forward(self, x):
        x = self.conv1(x)
        x_bn1 = self.bn1(x)
        x_act = self.activation(x_bn1)
        x_conv2,pred_map = self.conv2(x_act)
        # x_conv2 = self.bn2(x_conv2)
        output = x_conv2
        return output,pred_map


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class UNet_EF(nn.Module):
    """早期融合法U-Net"""

    def __init__(self, num_classes=2, input_channels=3):
        super().__init__()

        size = [256, 128, 64, 32]
        nb_filter = [32, 64, 128, 256, 512]
        status= [True, True, True, False, False]

        self.Maxpool = nn.MaxPool2d(2, 2)
        self.Avgpool = nn.AvgPool2d(2, 2)
        self.attn = SpatialAttention()
        self.conv0_0 = ConvIDCBlock(input_channels * 2, nb_filter[0], nb_filter[0], status[0])
        self.conv1_0 = ConvIDCBlock(nb_filter[1], nb_filter[1], nb_filter[1], status[1])
        self.conv2_0 = ConvIDCBlock(nb_filter[2], nb_filter[2], nb_filter[2], status[2])
        self.conv3_0 = ConvIDCBlock(nb_filter[3], nb_filter[3], nb_filter[3], status[3])
        self.conv4_0 = ConvIDCBlock(nb_filter[4], nb_filter[4], nb_filter[4], status[4])

        self.pappm0 = PAPPM(nb_filter[0], nb_filter[0]//2, nb_filter[0], 5)
        self.pappm1 = PAPPM(nb_filter[1], nb_filter[1]//2, nb_filter[1], 4)
        self.pappm2 = PAPPM(nb_filter[2], nb_filter[2]//2, nb_filter[2], 3)
        self.pappm3 = PAPPM(nb_filter[3], nb_filter[3]//2, nb_filter[3], 2)
        self.pappm4 = PAPPM(nb_filter[4], nb_filter[4]//2, nb_filter[4], 1)

        self.conv3_1 = RelationAwareFusion(channels=nb_filter[3])
        self.conv2_2 = RelationAwareFusion(channels=nb_filter[2])
        self.conv1_3 = RelationAwareFusion(channels=nb_filter[1])
        self.conv0_4 = RelationAwareFusion(channels=nb_filter[0])
        # self.conv_end = ConvBlock(nb_filter[1], nb_filter[0], nb_filter[0])

        # self.convd_end = RelationAwareFusion(channels=nb_filter[0], ext=1)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xA, xB):
        # def forward(self, xA):
        #     xB = xA
        x = torch.cat((xA, xB), dim=1)
        x0_0,pred_0 = self.conv0_0(x)

        x1_0,pred_1 = self.conv1_0(torch.cat((self.Maxpool(x0_0),self.Avgpool(x0_0)), dim=1))
        x2_0,pred_2 = self.conv2_0(torch.cat((self.Maxpool(x1_0),self.Avgpool(x1_0)), dim=1))
        x3_0,pred_3 = self.conv3_0(torch.cat((self.Maxpool(x2_0),self.Avgpool(x2_0)), dim=1))
        # x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0,pred_4 = self.conv4_0(torch.cat((self.Maxpool(x3_0),self.Avgpool(x3_0)), dim=1))


        refined_x4_0 = self.pappm4(x4_0)

        x3_0 = self.pappm3(x3_0)
        x3_1 = self.conv3_1(x3_0, refined_x4_0, True)
        x2_0 = self.pappm2(x2_0)
        x2_2 = self.conv2_2(x2_0, x3_1, True)
        x1_0 = self.pappm1(x1_0)
        x1_3 = self.conv1_3(x1_0, x2_2, True)
        x0_0 = self.pappm0(x0_0)
        x0_4 = self.conv0_4(x0_0, x1_3, True)

        # output = self.final(self.convd_end(xd, x0_4))
        # x_end = self.conv_end(torch.cat((x0_4, x0_4 * xd_attn), dim=1))
        output = self.final(x0_4)
        # output = self.final(x_end)
        # output = self.final(x_end)

        return pred_0,pred_1,pred_2,output


if __name__ == "__main__":
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    model = UNet_EF(num_classes=2, input_channels=3)
    flops, params = profile(model, inputs=(x1,x2))

    flops_in_million = flops / 1e6  # 将 FLOPs 转换为百万
    params_in_million = params / 1e6  # 将参数量转换为百万

    print(f"FLOPs: {flops_in_million:.2f} M")
    print(f"Parameters: {params_in_million:.2f} M")
    # model1 = UNet_EF()
    # flops, paras = get_model_complexity_info(model1, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    # print('flops:', flops, 'params:', paras)
