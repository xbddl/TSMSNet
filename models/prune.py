import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from models.networks import *
import torch.optim as optim
from models.losses import cross_entropy,deep_supervised_ce,Focal,edge_aware_loss,deep_supervised_edge_loss,deep_supervised_NR_Dice_loss,deep_supervised_fuzz_loss
from misc.metric_tool import ConfuseMatrixMeter
import os
from misc.logger_tool import Logger, Timer
from PIL import Image
from datasets.data_utils import CDDataAugmentation
from thop import profile

class CDPrune():
    def __init__(self, args, dataloaders, eval_dataloader):
        self.n_class = args.n_class
        self.dataloaders = dataloaders
        self.eval_dataloader = eval_dataloader
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        self.lr = args.prune_lr
        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define optimizers
        self.optimizer_G = None
        self.epochs = args.prune_epochs
        self.pruned_model = None
        self.checkpoint_dir = args.checkpoint_dir
        self.prune_checkpoint_dir = args.prune_checkpoint_dir
        self.pred_0 = None
        self.pred_1 = None
        self.pred_2 = None
        self.pred_3 = None
        self.pred_4 = None
        self.G_pred = None
        self.gt = None
        self.edge = None
        self.prune_best_iou = 0
        self.epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.prune_epochs
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch
        # define logger file
        logger_path = os.path.join(args.prune_checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        loss_logger_path = os.path.join(args.prune_checkpoint_dir, 'loss.txt')
        self.loss_logger = Logger(loss_logger_path)
        
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size
        
        # define the loss functions
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'deep_supervised_ce':
            self._pxl_loss = deep_supervised_ce
        elif args.loss == 'focal':
            self._pxl_loss = Focal
        elif args.loss == 'edge':
            self._pxl_loss = deep_supervised_edge_loss
        elif args.loss == 'NR_Dice_CE_Loss':
            self._pxl_loss = deep_supervised_NR_Dice_loss
        elif args.loss == 'fuzz':
            self._pxl_loss = deep_supervised_fuzz_loss
        else:
            raise NotImplemented(args.loss)
        
    def _load_checkpoint(self, ckpt_name='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading best checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
    
    def _save_checkpoint(self, model,ckpt_name):
        
        torch.save({
            'prune_model': model
        }, os.path.join(self.prune_checkpoint_dir, ckpt_name))
        
    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est
    
    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()

        
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.eval_dataloader)

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            if self.is_training is True:
                message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                        (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                        imps*self.batch_size, est,
                        self.G_loss.item(), running_acc)
            else:
                message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)
    
    def _collect_epoch_states(self,is_validation=False):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
            if k == 'precision_1' and is_validation == True:
                self.validate_precision = v
            if k == 'recall_1' and is_validation == True:
                self.validate_recall = v
        self.logger.write(message+'\n')
        self.logger.write('\n')
        return scores['iou_1']
    
    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = self.batch['A'].to(self.device)
        img_in2 = self.batch['B'].to(self.device)
        self.gt = self.batch['L'].to(self.device).float()
        self.edge = self.batch['E'].to(self.device).float()
        
        self.pred_0,self.pred_1,self.pred_2,self.G_pred = self.net_G(img_in1, img_in2)
        
    def _backward_G(self):
        
        self.G_loss = self._pxl_loss(self.pred_0, self.pred_1, self.pred_2, self.G_pred, self.gt, self.edge)
        self.G_loss.backward()

    def fine_tune(self):
        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=5e-4)
        best_iou = 0
        
        for self.epoch_id in range(self.epochs):
            self.is_training = True
            self.net_G.train()
            for batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)
                
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                #clip_grad_norm_(self.net_G.parameters(), max_norm=1.0)
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()

            iou = self._collect_epoch_states(False)
            val_iou = self.eval_models()
            
            if val_iou > best_iou:
                best_iou = val_iou
                self._save_checkpoint(self.net_G, 'best_prune.pth')
        self.prune_best_iou = best_iou

    def _clear_cache(self):
        self.running_metric.clear()

    def eval_models(self):

        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()
        for self.batch_id, batch in enumerate(self.eval_dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
            #self._save_deep_pred()
        iou = self._collect_epoch_states()
        return iou

    def tune(self):
        self._load_checkpoint()
        print("模型加载完成")
        best_iou = self.eval_models()
        print("best iou: %.5f", best_iou)
        augm = CDDataAugmentation(img_size=256)

        
        A_path = './example/A/34.png'
        B_path = './example/B/34.png'
        Label_path = './example/label/34.png'
        egde_path = './example/G3x3_edge_label/34.png'

        img_A = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        label = np.array(Image.open(Label_path), dtype=np.uint8)
        edge_label = np.array(Image.open(egde_path), dtype=np.uint8)

        label = label // 255
        edge_label = edge_label//255
        [img_A, img_B], [label], [edge_label] = augm.transform([img_A, img_B], [label], [edge_label], to_tensor=True)
        # 扩充batch 维度
        img_A = img_A.unsqueeze(0)
        img_B = img_B.unsqueeze(0)
        label = label.unsqueeze(0)
        edge_label = edge_label.unsqueeze(0)

        img_A = img_A.to(self.device)
        img_B = img_B.to(self.device)
        target = label.to(self.device).float()
        edge_target = edge_label.to(self.device).float()
        example_inputs = (img_A, img_B)
         
        bit_flops, bit_params = profile(self.net_G, inputs=(img_A, img_B), verbose=False)
        print(f"[剪枝前] FLOPs: {bit_flops/1e6:.2f} M, Params: {bit_params/1e6:.2f} M")
                
        from modules.new2_multi_sa.RAF_Pag  import  RelationAwareFusion
        from modules.new2_multi_sa.Multi_Modulation import PredictMap
        ignored_layers = []
        for m in self.net_G.modules():
            if isinstance(m, RelationAwareFusion):
                ignored_layers.append(m)
            if isinstance(m, PredictMap):
                ignored_layers.append(m)
        ignored_layers.append(self.net_G.final)
        

        
        num_iterations = 5
        imp = tp.importance.TaylorImportance()
        pruner = tp.pruner.MagnitudePruner(
            model=self.net_G,
            example_inputs=example_inputs,
            importance=imp,
            iterative_steps=5,
            pruning_ratio=0.5,
            global_pruning=True,
            ignored_layers=ignored_layers,
        )

        print("开始剪枝")
        tolerance = 0.01
        
        for i in range(num_iterations):
            if isinstance(imp, tp.importance.TaylorImportance):
                # loss = F.cross_entropy(model(images), targets)
                pred_0,pred_1,pred_2,G_pred = self.net_G(*example_inputs)
                loss = self._pxl_loss(pred_0, pred_1, pred_2, G_pred, target)
                loss.backward()
            pruner.step()
            self.net_G.zero_grad()
            self.fine_tune()
            print("best iou: %.5f, current iou: %.5f", best_iou, self.prune_best_iou)
            bit_flops, bit_params = profile(self.net_G, inputs=(img_A, img_B), verbose=False)
            print(f"[剪枝后] FLOPs: {bit_flops/1e6:.2f} M, Params: {bit_params/1e6:.2f} M")
            pth_name = 'prune_' + str(i) + '_iters_FLOPs_' + str(bit_flops/1e6) + '_M_Params_' + str(bit_params/1e6) + '_M_IoU_' + str(self.prune_best_iou) + '.pth'
            self._save_checkpoint(self.net_G, pth_name)
            if self.prune_best_iou < best_iou - tolerance:
                print("性能下降，停止剪枝")
                break
            
            

      
