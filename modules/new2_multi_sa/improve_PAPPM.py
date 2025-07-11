import torch
import torch.nn as nn
import torch.nn.functional as F

class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, scales, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scales = scales
        self.pooling = nn.ModuleList([nn.Sequential(nn.AvgPool2d(kernel_size=3 + i * 4, stride=1 + i * 2, padding=1 + i * 2),
                                                    BatchNorm(inplanes, momentum=bn_mom),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
                                     for i in range(scales)])
        
        self.GAP = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)),
                                BatchNorm(inplanes, momentum=bn_mom),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))

        
        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        
        self.integration = nn.Sequential(
            BatchNorm(branch_planes * (scales + 1), momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * (scales + 1), branch_planes, kernel_size=1, bias=False)
        )
        
        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 2, outplanes, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )
        
        self.att = channel_attention(branch_planes)

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)

        for i in range(self.scales):
            scale_list.append(F.interpolate(self.pooling[i](x), size=(height, width), mode='bilinear', align_corners=True) + x_)
        scale_list.append(F.interpolate(self.GAP(x), size=(height, width), mode='bilinear', align_corners=True) + x_)
        

        scale_out = self.integration(torch.cat(scale_list, 1))
        
        scale_out = self.att(scale_out)
        
        scale_out = self.compression(torch.cat((scale_out, x_), 1))

        out = scale_out + self.shortcut(x)
        return out

class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(channel_attention, self).__init__()
        self.GAP = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False),
                                )
        self.point_wise = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False),
                                        )
        self.local_attention = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False),
                                            )
        self.sigmoid = nn.Sigmoid()
        self.shortcut = nn.Identity()
        
    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        shortcut = self.shortcut(x)
        ch_att = self.GAP(x) + self.point_wise(x)
        ch_att += F.interpolate(self.local_attention(x), size=(height, width), mode='bilinear', align_corners=True)
        return x * self.sigmoid(ch_att) + shortcut
        

if __name__ == "__main__":
    x = torch.randn(4, 512, 32, 32).cuda()
    model1 = PAPPM(inplanes=512, branch_planes=512, outplanes=512).cuda()
    output1 = model1(x).cuda()
    print(output1.size())
