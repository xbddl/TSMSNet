# ISDNet: Integrating Shallow and Deep Networks for Efficient Ultra-high Resolution Segmentation(CVPR2022)
# PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers( CVPR2023 )

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pag(nn.Module):
    def __init__(self, in_channels):
        super(Pag, self).__init__()
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 8)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 8)
        )
        # self.g = nn.Parameter(torch.zeros(1))
        
        

        
        #加入两个可学习的标量a,b,随机初始值
        self.a = nn.Parameter(torch.tensor(1.0,requires_grad=True))
        self.b = nn.Parameter(torch.tensor(1.0,requires_grad=True))
        self.conv = nn.Conv2d(2 * in_channels , in_channels, kernel_size=1)

    def forward(self, x, y):
        
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_avg = torch.mean(x, dim=1, keepdim=True)
        y_max = torch.max(y, dim=1, keepdim=True)[0]
        y_avg = torch.mean(y, dim=1, keepdim=True)
        
        x_k = torch.cat([x_max, x_avg], dim=1)
        y_k = torch.cat([y_max, y_avg], dim=1)
        # sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        # x = (1 - sim_map) * x + sim_map * y
        local_map = torch.sigmoid(torch.sum(x_k * y_k, dim=1).unsqueeze(1))
        x = self.a * local_map * x
        y = self.b * local_map * y
        
        output = x + y

        # x = self.g * x + y

        return output


class ChannelAtt(nn.Module):
    # def __init__(self, channels, out_channels, conv_cfg, norm_cfg, act_cfg):
    def __init__(self, channels, out_channels):
        super(ChannelAtt, self).__init__()
        # self.conv_bn_relu = ConvModule(channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
        #                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        # self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg,
        #                            norm_cfg=norm_cfg, act_cfg=None)
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(channels, out_channels, 3, 1, 1),
                                          nn.BatchNorm2d(out_channels),
                                          nn.LeakyReLU())
        self.conv_1x1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, 1, 0),
                                      nn.BatchNorm2d(out_channels))

    def forward(self, x, fre=False):
        """Forward function."""
        feat = self.conv_bn_relu(x)
        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, atten


class RelationAwareFusion(nn.Module):
    # def __init__(self, channels, conv_cfg, norm_cfg, act_cfg, ext=2, r=16):
    def __init__(self, channels, ext=2, r=32):
        super(RelationAwareFusion, self).__init__()
        self.r = r
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        # self.spatial_mlp = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.spatial_mlp = nn.Sequential(nn.Linear(r * r, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.spatial_att = ChannelAtt(channels, channels)
        # self.context_mlp = nn.Sequential(*[nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels)])
        self.context_mlp = nn.Sequential(*[nn.Linear(r * r, channels), nn.ReLU(), nn.Linear(channels, channels)])
        self.context_att = ChannelAtt(channels * ext, channels)
        # self.context_head = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
        #                                act_cfg=act_cfg)
        # self.smooth = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
        #                          act_cfg=None)
        self.context_head = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(channels),
                                          nn.ReLU())
        self.channel_align = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(channels),
                                          nn.ReLU())
        self.pag = Pag(channels)
        self.smooth = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(channels))


    def forward(self, sp_feat, co_feat, Identity=True):
        # **_att: B x C x 1 x 1
        # sp_feat(B,C1,H1,W1),co_feat(B,C2,H2,W2)
        
        s_feat, s_att = self.spatial_att(sp_feat)  # (B,C1/2,H1,W1),(B,C1/2,1,1)
        c_feat, c_att = self.context_att(co_feat)  # (B,C2,H2,W2),(B,C2,1,1)
        
        b, c, h, w = s_att.size()
        s_att_split = s_att.view(b, self.r, c // self.r)  # (B,r,C1/r)
        c_att_split = c_att.view(b, self.r, c // self.r)  # (B,r,C1/r)
        chl_affinity = torch.bmm(s_att_split, c_att_split.permute(0, 2, 1))  # (B,r,r)
        chl_affinity = chl_affinity.view(b, -1)  # (B,r*r)
        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity))  
        co_mlp_out = F.relu(self.context_mlp(chl_affinity))
        re_s_att = torch.sigmoid(s_att + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_c_att = torch.sigmoid(c_att + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1))
        s_feat = torch.mul(s_feat, re_s_att)
        c_feat = torch.mul(c_feat, re_c_att)
        c_feat = F.interpolate(c_feat, s_feat.size()[2:], mode='bilinear', align_corners=False)
        c_feat = self.context_head(c_feat)
        out1 = self.pag(s_feat, c_feat)
        
        out = self.smooth(out1)
        # return s_feat, c_feat, out
        return out


if __name__ == "__main__":
    x1 = torch.randn(4, 256, 32, 32)
    # x2 = torch.randn(4, 512, 16, 16)
    x2 = torch.randn(4, 256, 32, 32)
    model = RelationAwareFusion(channels=256, ext=1)
    out = model(x1, x2)
    print(out.size())
