import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils

import models.ConfusionMatrix as ConfusionMatrix
import models.wt2excel as wt2excel
import models.color as color

from contrast_models.contrast_network import define_G as define_contrast_G

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize_predict(predmap):
    if predmap.shape[1] == 2:
        pred = torch.argmax(predmap, dim=1, keepdim=True)
    elif predmap.shape[1] == 1:
        pred = torch.sigmoid(predmap)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
    pred_vis = pred * 255
    return pred_vis



class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        self.batch_size = args.batch_size
        self.data_name = args.data_name
        # define G
        self.BIT = define_contrast_G(net_G='BIT', gpu_ids=args.gpu_ids)
        self.MSCANet = define_contrast_G(net_G='MSCANet', gpu_ids=args.gpu_ids)
        self.SNUNet = define_contrast_G(net_G='SNUNet', gpu_ids=args.gpu_ids)
        self.IFN = define_contrast_G(net_G='IFN', gpu_ids=args.gpu_ids)
        self.DMINet = define_contrast_G(net_G='DMINet', gpu_ids=args.gpu_ids)
        self.STADE_CDNet = define_contrast_G(net_G='STADE_CDNet', gpu_ids=args.gpu_ids)
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        
        self.baseline = define_baseline(args=args, gpu_ids=args.gpu_ids)
        
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)
        self.target = None
        
        self.pred_0 = None
        self.pred_1 = None
        self.pred_2 = None
        
        self.baseline_pred_0 = None
        self.baseline_pred_1 = None
        self.baseline_pred_2 = None
        
        self.bit_output = None
        self.msca_output = None
        self.snunet_output = None
        self.ifn_output = None
        self.dminet_output = None
        self.stade_output = None
        
        self.pred_3 = None
        self.pred_4 = None
        self.G_pred = None
        self.baseline_pred = None
        
        self.pred_list = []
        self.check_list = []
        
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.baseline_checkpoint_dir = args.baseline_checkpoint_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.contrast_checkpoint_dir = 'checkpoints'
        self.vis_dir = args.vis_dir
        self.test_iou = []
        # check and create model dir
        if os.path.exists(self.baseline_checkpoint_dir) is False:
            os.mkdir(self.baseline_checkpoint_dir)
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_BIT(self, checkpoint_name ='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'BIT',checkpoint_name)):
            self.logger.write('loading BIT weight...\n')
            checkpoint = torch.load(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'BIT',checkpoint_name), map_location=self.device)
            self.BIT.load_state_dict(checkpoint['model_G_state_dict'])
            self.BIT.to(self.device)
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        
    def _load_MSCANet(self, checkpoint_name ='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'MSCANet',checkpoint_name)):
            self.logger.write('loading MSCANet weight...\n')
            checkpoint = torch.load(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'MSCANet',checkpoint_name), map_location=self.device)
            self.MSCANet.load_state_dict(checkpoint['model_G_state_dict'])
            self.MSCANet.to(self.device)
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _load_SNUNet(self, checkpoint_name ='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'SNUNet',checkpoint_name)):
            self.logger.write('loading SNUNet weight...\n')
            checkpoint = torch.load(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'SNUNet',checkpoint_name), map_location=self.device)
            self.SNUNet.load_state_dict(checkpoint['model_G_state_dict'])
            self.SNUNet.to(self.device)
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        
    def _load_DMINet(self, checkpoint_name ='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'DMINet',checkpoint_name)):
            self.logger.write('loading DMINet weight...\n')
            checkpoint = torch.load(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'DMINet',checkpoint_name), map_location=self.device)
            self.DMINet.load_state_dict(checkpoint['model_G_state_dict'])
            self.DMINet.to(self.device)
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        
    def _load_STADE_CDNet(self, checkpoint_name ='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'STADE_CDNet',checkpoint_name)):
            self.logger.write('loading STADE_CDNet weight...\n')
            checkpoint = torch.load(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'STADE_CDNet',checkpoint_name), map_location=self.device)
            self.STADE_CDNet.load_state_dict(checkpoint['model_G_state_dict'])
            self.STADE_CDNet.to(self.device)
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        
    def _load_IFN(self, checkpoint_name ='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'IFN',checkpoint_name)):
            self.logger.write('loading IFN weight...\n')
            checkpoint = torch.load(os.path.join(self.contrast_checkpoint_dir, self.data_name, 'IFN',checkpoint_name), map_location=self.device)
            self.IFN.load_state_dict(checkpoint['model_G_state_dict'])
            self.IFN.to(self.device)
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _set_eval(self):
        self.net_G.eval()
        self.BIT.eval()
        self.MSCANet.eval()
        self.SNUNet.eval()
        self.IFN.eval()
        self.DMINet.eval()
        self.STADE_CDNet.eval()
        
    def _load_contrast_model(self):
        self._load_BIT()
        self._load_MSCANet()
        self._load_SNUNet()
        self._load_IFN()
        self._load_DMINet()
        self._load_STADE_CDNet()
    
    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        G_pred = self.G_pred.detach()
        pred = torch.argmax(G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        #print("target.shape",target.shape)
        G_pred = self.G_pred.detach()
        #print("Gpred.shape",G_pred.shape)
        G_pred = torch.argmax(G_pred, dim=1)
        #print("Gpred.shape",G_pred.shape)
        #self.test_iou.append(self._collect_each_iou(target,G_pred))
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score



    def _collect_running_batch_states(self):

        running_acc = self._update_metric()
        self.check_list.append(self._check_is_top())

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        batch_size = self.batch['A'].shape[0]
        
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))

        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

        vis_pred = utils.make_numpy_grid(visualize_predict(self.G_pred.detach()))

        vis_bit = utils.make_numpy_grid(visualize_predict(self.bit_output.detach()))
        
        vis_msca = utils.make_numpy_grid(visualize_predict(self.msca_output.detach()))
        
        vis_snunet = utils.make_numpy_grid(visualize_predict(self.snunet_output.detach()))
        
        vis_ifn = utils.make_numpy_grid(visualize_predict(self.ifn_output.detach()))
        
        vis_dminet = utils.make_numpy_grid(visualize_predict(self.dminet_output.detach()))
        
        vis_stadenet = utils.make_numpy_grid(visualize_predict(self.stade_output.detach()))
        
        vis_gt = utils.make_numpy_grid(self.batch['L'])
        
        result_pred = color.compare_gt_pred(vis_gt,vis_pred)
        result_bit = color.compare_gt_pred(vis_gt,vis_bit)
        result_msca = color.compare_gt_pred(vis_gt,vis_msca)
        result_snunet = color.compare_gt_pred(vis_gt,vis_snunet)
        result_ifn = color.compare_gt_pred(vis_gt,vis_ifn)
        result_dminet = color.compare_gt_pred(vis_gt,vis_dminet)
        result_stadenet = color.compare_gt_pred(vis_gt,vis_stadenet)

        # 假设所有输入数组都有相同的形状 (h, b*w, c)  
        h, bw, c = vis_input.shape  
        w = bw // batch_size  # 假设 b 等于 self.batch_size，因此每个大图像包含 self.batch_size 个小图像  
        
        # 灰色间隔的颜色  
        gray_color = [0.5, 0.5, 0.5] if c == 3 else 0.5  
        
        # 计算最终图像的宽度和高度  
        # 假设我们想要将所有批次的所有小图像平铺到一个大的图像中  
        # 每个小图像之间有一个间隔，批次之间也有一个垂直间隔  
        spacing = 10  # 间隔宽度（也适用于垂直间隔）  
        num_small_images_per_row = batch_size  # 每行的小图像数量  
        num_rows = len([vis_input, vis_input2, vis_gt, result_pred, result_bit, result_msca, result_snunet, result_ifn, result_dminet, result_stadenet])  # 批次数量（这里假设每个数组都是一个批次）  
        
        # 计算大图像的尺寸  
        total_width = num_rows * (h + spacing) - spacing
        total_height =   num_small_images_per_row * (w + spacing) - spacing  
        
        # 创建一个空白的大图像  
        combined_image = np.full((total_height, total_width, c), gray_color, dtype=np.float32)  
        
        # 填充大图像  
        current_x = 0  # 水平位置 
        for batch_idx, batch_array in enumerate([vis_input, vis_input2, vis_gt, result_pred, result_bit, result_msca, result_snunet, result_ifn, result_dminet, result_stadenet]):  
            current_y = 0  # 垂直位置  
            for small_img_idx in range(batch_size):  # 遍历每个小图像  
                # 提取当前小图像（注意：这里假设 b 等于 self.batch_size）  
                small_img = batch_array[:, small_img_idx*w:(small_img_idx+1)*w, :]  
                
                # 将小图像复制到组合图像中  
                combined_image[current_y:current_y+h, current_x:current_x+w, :] = small_img  
                
                # 更新水平位置以包含间隔  
                current_y += h + spacing  
                 
            
            # 更新垂直位置以包含间隔（在移动到下一个批次之前）  
            current_x += w + spacing 
        
        # 确保值在 [0, 1] 范围内（尽管直接复制应该已经满足这个条件，但以防万一）  
        combined_image = np.clip(combined_image, 0.0, 1.0)  
        
        # 显示或保存大图像  
        # plt.imshow(combined_image)  
        # plt.axis('off')  # 不显示坐标轴  
        # plt.show()  

        # vis = np.concatenate([vis_input, vis_input2, vis_gt, result_pred,result_bit,result_msca,result_snunet,result_ifn,result_dminet,result_stadenet], axis=0)
        # vis = np.clip(vis, a_min=0.0, a_max=1.0)

        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        plt.imsave(file_name, combined_image)
        
    def _calculate_single_map_iou(self,pred,gt,class_num=2):
        class_num = self.n_class
        metric = ConfuseMatrixMeter(class_num)
        hist = metric.update_cm(pred, gt)
        score = metric.get_scores()
        Iou = score['iou_1']
        return Iou
    
    def _collect_each_iou(self,target,G_pred):
        batch_size = self.batch_size
        
        iou_list = []
        for i in range(batch_size):
            pred = G_pred[i:i+1]
            #print(pred.shape)
            gt = target[i:i+1]
            iou = self._calculate_single_map_iou(pred.cpu().numpy(),gt.cpu().numpy())
            iou_list.append(iou)
            
        return iou_list
        
    def _check_is_top(self):
        batch_size = self.batch_size
        target = self.target.detach()
        pred_list = []
        label_list = []
        
        
        for k in range(len(self.pred_list)):
            pred = self.pred_list[k]
            pred = pred.detach()
            if pred.shape[1] == 1:
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).int().to(torch.int64)
            else:
                pred = torch.argmax(pred, dim=1)
            pred_list.append(pred)
        
        for i in range(batch_size):
            iou_list = []
            for k in range(len(pred_list)):
                pred = pred_list[k]
                pred = pred[i:i+1]
                gt = target[i:i+1]
                iou = self._calculate_single_map_iou(pred.cpu().numpy(),gt.cpu().numpy())
                iou_list.append(iou)
            if iou_list[0] == max(iou_list):
                label_list.append(1)
            else:
                label_list.append(0)
        return label_list

            

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        self.target = batch['L'].to(self.device)
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        _,_,_,self.G_pred = self.net_G(img_in1, img_in2)
        self.bit_output = self.BIT(img_in1, img_in2)
        self.msca_output,_,_ = self.MSCANet(img_in1, img_in2)
        self.snunet_output, = self.SNUNet(img_in1, img_in2)
        self.ifn_output,_,_,_,_ = self.IFN(img_in1, img_in2)
        self.pred_0,self.pred_1,self.pred_2,self.pred_3 = self.DMINet(img_in1, img_in2)
        self.dminet_output = self.pred_0 + self.pred_1
        self.stade_output = self.STADE_CDNet(img_in1, img_in2)[-1]
        self.pred_list = [self.G_pred, self.bit_output, self.msca_output, self.snunet_output, self.ifn_output, self.dminet_output, self.stade_output]
        


    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)
        self._load_contrast_model()
        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self._set_eval()
        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
            #self._save_deep_pred()
        wt2excel.write_list_to_excel(self.check_list, 'vis/check_list.xlsx')
        self._collect_epoch_states()
