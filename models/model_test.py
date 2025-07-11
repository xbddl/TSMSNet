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
        self.net_G = None
        
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
        
        self.pred_3 = None
        self.pred_4 = None
        self.G_pred = None
        self.baseline_pred = None
        
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.test_iou = []

    def _load_model(self, checkpoint_name = 'best_prune.pth'):
        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading best checkpoint...\n')
            # load model & weight
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)
            self.net_G = checkpoint['prune_model']
            self.net_G.to(self.device)
            self.logger.write('load best checkpoint done!\n')


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
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score



    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

    
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

    def eval_models(self,checkpoint_name='best_prune.pth'):

        self._load_model(checkpoint_name)
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
        self._collect_epoch_states()
