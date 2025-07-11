from argparse import ArgumentParser
import torch
from models.prune import *
import utils

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""


def Net_prune(args):
    
    eval_dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    train_dataloaders = utils.get_loaders(args)
    netprune = CDPrune(args, train_dataloaders, eval_dataloader)
    netprune.tune()


 

if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='HGG_best', type=str)
    parser.add_argument('--checkpoint_root', default='new1', type=str)  #结果的保存路径
    parser.add_argument('--baseline_checkpoint_dir', default='baseline', type=str)  #基线网络的保存路径

    # data
    parser.add_argument('--num_workers', default=0, type=int)  
    parser.add_argument('--dataset', default='CDDataset', type=str) #定义数据集的名称
    parser.add_argument('--data_name', default='HGG', type=str) #数据集的路径,位于utils.py文件下的import data_config内部
    # parser.add_argument('--data_name', default='quick_start', type=str)

    parser.add_argument('--batch_size', default=8, type=int)   #每次迭代的图片数
    parser.add_argument('--split', default="train", type=str)  #训练文件名
    # parser.add_argument('--split_val', default="val", type=str) #验证的文件名
    parser.add_argument('--split_val', default="val", type=str) #验证的文件名

    parser.add_argument('--img_size', default=256, type=int)  #图片大小

    # model
    parser.add_argument('--n_class', default=2, type=int)
    # parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8', type=str,
    #                     help='base_resnet18 | base_transformer_pos_s4 | '
    #                          'base_transformer_pos_s4_dd8 | '
    #                          'base_transformer_pos_s4_dd8_dedim8|')
    # parser.add_argument('--net_G', default='unet_base', type=str)  #网络的名称
    parser.add_argument('--net_G', default='unet_base', type=str)  #网络的名称
    
    parser.add_argument('--baseline', default='unet_base', type=str)  #网络的名称
    # parser.add_argument('--loss', default='ce', type=str)  #损失函数的选取
    parser.add_argument('--loss', default='fuzz', type=str)  #损失函数的选取

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)  #优化器
    parser.add_argument('--lr', default=0.01, type=float)   #初始学习率
    parser.add_argument('--prune_lr', default=0.001, type=float)   #初始学习率
    parser.add_argument('--max_epochs', default=200, type=int)   #最大轮数
    parser.add_argument('--prune_epochs', default=20, type=int)   #最大轮数
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step | multistep')     #学习率更新策略,位于models文件夹的networks.py的def get_scheduler函数下面
    parser.add_argument('--lr_decay_iters', default=100, type=int)  #学习率的衰减轮数

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    args.prune_checkpoint_dir = os.path.join('prune', args.project_name)
    os.makedirs(args.prune_checkpoint_dir, exist_ok=True)
    
    Net_prune(args)
