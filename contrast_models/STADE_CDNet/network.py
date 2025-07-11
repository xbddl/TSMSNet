import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import functools
from einops import rearrange
import cv2
from .help_funcs import *

import contrast_models.STADE_CDNet.resnet as models
# from .resnet import resnet18, resnet34, resnet50

device = "cuda" if torch.cuda.is_available() else "cpu"   


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x

class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)
 
    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
                                #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
                                #   self.hidden_dim)).cuda()
        x = x.to(device)
        out, _ = self.lstm(x)
        
        return out

class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x
    def zl_difference(self, x):
        # # #zl_CDDM###############
       
        ##1 step
        avg_pooling = torch.nn.AdaptiveMaxPool2d((256, 256))
        _c1_21 = avg_pooling(x)
        _c1_21 = rearrange(_c1_21, 'b c h w -> b c (h w)')
        ##1.1 step
        conv = nn.Conv1d(_c1_21.shape[1], x.shape[2], kernel_size=3, padding=1, groups=1, stride=1, bias=False).cuda()
        _c1_22=conv(_c1_21)   
        nn.ReLU(),
        ##1.2 step
        conv1_LZ = nn.Conv1d(x.shape[2], x.shape[1], kernel_size=3, padding=1, groups=1, stride=1, bias=False).cuda()
        _c1_24=conv1_LZ(_c1_22)
        _c1_24=_c1_24.unsqueeze(0)
        #_c1_24=np.transpose(_c1_24.numpy(),(1,2,3,0)).cuda()
        _c1_24_transposed_gpu = _c1_24.permute(1, 2, 3, 0)
        nn.Sigmoid(),
        _c1_24=_c1_24.squeeze(3)
        _c1_24 = x.reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3],)
        #M_1J
        _c1_25_M_1J=_c1_24*x
        #2 step
        #2.1 step 
        conv3_LZ = nn.Conv3d( _c1_25_M_1J.shape[0],_c1_25_M_1J.shape[0], kernel_size=3, padding=1, groups=1, stride=1, bias=False).cuda()
        _c1_26_M_1J=conv3_LZ(_c1_25_M_1J)
        nn.Sigmoid(),
        #2.2 step
        conv3_LZ = nn.Conv3d(_c1_26_M_1J.shape[0], _c1_25_M_1J.shape[0], kernel_size=3, padding=1, groups=1, stride=1, bias=False).cuda()
        _c1_27_M_1J=conv3_LZ(_c1_26_M_1J)
        #M_2J
        _c1_28_M_2J=_c1_27_M_1J*_c1_25_M_1J
        #result
        _c1_29_M_2J =_c1_28_M_2J.squeeze(0)
        out=_c1_29_M_2J
        # ###CDDM###############

        return out
    def forward(self, x1, x2):
        # forward backbone resnet
        x_n0=abs(x1-x2)
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
       
        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        x101 = abs(x1-x2)
        if not self.if_upsample_2x:
            x101 = self.upsamplex2(x101)
        x101 = self.upsamplex4(x101)
        # forward small cnn
        x101 = self.classifier(x101)


        x11=x1
        if not self.if_upsample_2x:
            x11 = self.upsamplex2(x11)
        x11 = self.upsamplex4(x11)
        # forward small cnn
        x110 = self.classifier(x11)

        x22=x2
        if not self.if_upsample_2x:
            x22 = self.upsamplex2(x22)
        x22 = self.upsamplex4(x22)
        # forward small cnn
        x22 = self.classifier(x22)

     #TMM
        model = Rnn(32, 32, 2, 2)
        model = model.cuda()
          #Maxpooling
        maxPool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, dilation=1,return_indices=False, ceil_mode=False)
        ###fusion
        
        
        x_add=x1+x2
        x_maxpool=maxPool( x_add)
        x_before = self.upsamplex2(x_maxpool)
        x_after = rearrange(x_before, 'b c h w -> b (h w) c')
        x_TMM_0 = model(x_after)
        x_TMM_1 = x_TMM_0 .reshape(x101.shape[0],x101.shape[1],x101.shape[2],x101.shape[3],)
        
     #CDDM
        x111=self.zl_difference(x110)
       
        x222=self.zl_difference(x22)
       
        
        x_CDDM=torch.abs( x111- x222 )
        if not self.if_upsample_2x:
            x_CDDM = self.upsamplex2(x_CDDM)
        
       # forward small cnn
        
        x=x_CDDM*x101+x101+x_TMM_1
      
        if self.output_sigmoid:
            x = self.sigmoid(x)
        outputs = []
        outputs.append(x)
        return outputs