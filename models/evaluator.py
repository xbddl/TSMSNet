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

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        self.batch_size = args.batch_size
        # define G
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

        self.pred_0 = None
        self.pred_1 = None
        self.pred_2 = None
        
        self.baseline_pred_0 = None
        self.baseline_pred_1 = None
        self.baseline_pred_2 = None
        
        self.pred_3 = None
        self.pred_4 = None
        self.G_pred = None
        self.baseline_pred = None
        
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.baseline_checkpoint_dir = args.baseline_checkpoint_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.test_iou = []
        # check and create model dir
        if os.path.exists(self.baseline_checkpoint_dir) is False:
            os.mkdir(self.baseline_checkpoint_dir)
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


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
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis
    
    def _visualize_deep_pred(self):
        pred0 = torch.argmax(self.pred_0, dim=1, keepdim=True)
        pred0 = pred0 * 255
        
        
        pred1 = torch.argmax(self.pred_1, dim=1, keepdim=True)
        pred1 = pred1 * 255
        
        
        pred2 = torch.argmax(self.pred_2, dim=1, keepdim=True)
        pred2= pred2 * 255
        
        
        return pred0, pred1, pred2
    
# 加载基线网络参数        
    def _load_baseline_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.baseline_checkpoint_dir, checkpoint_name)):
            self.logger.write('loading baseline last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.baseline_checkpoint_dir, checkpoint_name), map_location=self.device)
            
            self.baseline.load_state_dict(checkpoint['model_G_state_dict'])
            
            self.baseline.to(self.device)
            
            # update some other states
            self.baseline_best_val_acc = checkpoint['best_val_acc']
            self.baseline_best_epoch_id = checkpoint['best_epoch_id']
            
            self.logger.write('Eval Baseline Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.baseline_best_val_acc, self.baseline_best_epoch_id))
            self.logger.write('\n')
            
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _visualize_baseline_pred(self):
        pred = torch.argmax(self.baseline_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _visualize_daseline_deep_pred(self):
        pred0 = torch.argmax(self.baseline_pred_0, dim=1, keepdim=True)
        pred0 = pred0 * 255
        
        
        pred1 = torch.argmax(self.baseline_pred_1, dim=1, keepdim=True)
        pred1 = pred1 * 255
        
        
        pred2 = torch.argmax(self.baseline_pred_2, dim=1, keepdim=True)
        pred2= pred2 * 255
        
        
        return pred0, pred1, pred2

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
        self.test_iou.append(self._collect_each_iou(target,G_pred))
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score


    def _save_deep_pred(self):
        
        vis_gt = utils.make_numpy_grid(self.batch['L'])
        pred0, pred1, pred2 = self._visualize_deep_pred()
        baseline_pred0, baseline_pred1, baseline_pred2 = self._visualize_daseline_deep_pred()
        
        vis = np.concatenate([pred0.cpu(), baseline_pred0.cpu()], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_deep_pred_' + str(self.batch_id)+'.jpg')
        plt.imsave(file_name, vis)

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))

        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

        vis_pred = utils.make_numpy_grid(self._visualize_pred())

        vis_baseline_pred = utils.make_numpy_grid(self._visualize_baseline_pred())

        
        vis_gt = utils.make_numpy_grid(self.batch['L'])
        
        result_pred = color.compare_gt_pred(vis_gt,vis_pred)
        result_baseline_pred = color.compare_gt_pred(vis_gt,vis_baseline_pred)

        vis = np.concatenate([vis_input, vis_input2, result_pred, result_baseline_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        plt.imsave(file_name, vis)
        
        pred0, pred1, pred2 = self._visualize_deep_pred()
        
        pred0 = utils.make_numpy_grid(pred0)
        pred1 = utils.make_numpy_grid(pred1)
        pred2 = utils.make_numpy_grid(pred2)

        
        vis = np.clip(pred0, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_deep_pred_0_' + str(self.batch_id)+'.jpg')
        plt.imsave(file_name, vis)
        
        vis_1 = np.clip(pred1, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_deep_pred_1_' + str(self.batch_id)+'.jpg')
        plt.imsave(file_name, vis_1)
        
        vis_2 = np.clip(pred2, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_deep_pred_2_' + str(self.batch_id)+'.jpg')
        plt.imsave(file_name, vis_2)
        
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
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.pred_0,self.pred_1,self.pred_2,self.G_pred = self.net_G(img_in1, img_in2)
        self.baseline_pred_0,self.baseline_pred_1,self.baseline_pred_2,self.baseline_pred = self.baseline(img_in1, img_in2)

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)
        self._load_baseline_checkpoint(checkpoint_name)
        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
            #self._save_deep_pred()
        wt2excel.write_list_to_excel(self.test_iou, 'vis/iou_list.xlsx')
        self._collect_epoch_states()
