from thop import profile
import torch

from BIT.BIT import BASE_Transformer as BIT
from CropLand.network import CDNet
from SNUNet.Models import SNUNet_ECAM
from IFN.DSIFN import DSIFN
from DMINet.DMINet import DMINet
from STADE_CDNet.network import BASE_Transformer as STADE_CDNet

device = "cuda" if torch.cuda.is_available() else "cpu"   
bit = BIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8).to(device)
MSCANet = CDNet(img_size=256).to(device)
snu = SNUNet_ECAM(3,2).to(device)
ifn = DSIFN(use_dropout=True).to(device)
dmi = DMINet(pretrained=True).to(device)
stade = STADE_CDNet(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8).to(device)

x1 = torch.randn(1, 3, 256, 256).to(device)
x2 = torch.randn(1, 3, 256, 256).to(device)

bit_flops, bit_params = profile(bit, inputs=(x1, x2))
mscanet_flops, mscanet_params = profile(MSCANet, inputs=(x1, x2))
snu_flops, snu_params = profile(snu, inputs=(x1, x2))
ifn_flops, ifn_params = profile(ifn, inputs=(x1, x2))
dminet_flops, dminet_params = profile(dmi, inputs=(x1, x2))
stade_flops, stade_params = profile(stade, inputs=(x1, x2))


bit_params = bit_params / 1e6
mscanet_params = mscanet_params / 1e6
snu_params = snu_params / 1e6
ifn_params = ifn_params / 1e6
dminet_params = dminet_params / 1e6
stade_params = stade_params / 1e6

bit_flops = bit_flops / 1e9
mscanet_flops = mscanet_flops / 1e9
snu_flops = snu_flops / 1e9
ifn_flops = ifn_flops / 1e9
dminet_flops = dminet_flops / 1e9
stade_flops = stade_flops / 1e9

print('BIT params: {:.2f}M, flops: {:.2f}G'.format(bit_params, bit_flops))
print('MSCANet params: {:.2f}M, flops: {:.2f}G'.format(mscanet_params, mscanet_flops))
print('SNUNet params: {:.2f}M, flops: {:.2f}G'.format(snu_params, snu_flops))
print('IFN params: {:.2f}M, flops: {:.2f}G'.format(ifn_params, ifn_flops))
print('DMINet params: {:.2f}M, flops: {:.2f}G'.format(dminet_params, dminet_flops))
print('STADE_CDNet params: {:.2f}M, flops: {:.2f}G'.format(stade_params, stade_flops))

