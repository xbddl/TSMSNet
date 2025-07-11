from torch.utils.tensorboard import SummaryWriter
import torch
import os
import numpy as np
from PIL import Image
from datasets.data_utils import CDDataAugmentation
from modules.new2_multi_sa.new2 import UNet_EF as new2


gpu_ids = ['0']
device = torch.device("cuda:%s" % gpu_ids[0] if torch.cuda.is_available() and len(gpu_ids)>0
                                   else "cpu")
dummy_input = (torch.randn(1,3,256,256).cuda(), torch.randn(1,3,256,256).cuda())

checkpoint_dir = './prune/LEVIR_best'
checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_prune.pth'), map_location=device)
net_G = checkpoint['prune_model']
net_G.eval().cuda()
torch.onnx.export(net_G, dummy_input, "pruned_model_LEVIR.onnx", 
                  opset_version=11, verbose=True, input_names=['t1','t2'],
                  output_names=['output'])

