import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from ipdb import set_trace as st

from math import sqrt
from tqdm import tqdm
from .train_model import TrainModel
from networks import networks

import util.pytorch_ssim as pytorch_ssim

from sklearn.metrics import confusion_matrix
import util.semseg.metrics.raster as metrics
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_ssim
from skimage import measure
# import sobel
# from piqa import SSIM
from torch import optim
# from sewar.full_ref import ssim
# import pytorch_ssim
# import torch
import torch.nn.functional as F

#from piq import ssim, SSIMLoss

#from piqa import SSIM

#class SSIMLoss(SSIM):
#    def forward(self, x, y):
#        return 1. - super().forward(x, y)
#def L_blur(images):
 #   beta = 2.5 # or any other value of your choice
  #  N = images.shape[0]
   # M = images.shape[1]*images.shape[2]
    #mu = images.mean()
    #loss = 0.0
    #for c in range(N):
           
     #   Lap_filter = torch.Tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
      #  dX = F.conv2d(images[c,:,:], Lap_filter, padding=1)
       # loss -= beta * torch.log(torch.sum(dX**2) / (M - mu**2))
   # blurr_loss /= N
    #return blurr_loss



# Be able to add many loss functions
class MultiTaskGen(TrainModel):
    def name(self):
        return 'MultiTask General Model'

    def initialize(self, opt):
        TrainModel.initialize(self, opt)
        
        if self.opt.resume:
            self.netG, self.optimG = self.load_network()
        elif self.opt.train:
            from os.path import isdir
            if isdir(self.opt.pretrained_path) and self.opt.pretrained:
                self.netG = self.load_weights_from_pretrained_model()
            else:
                self.netG = self.create_network()
            self.optimG = self.get_optimizerG(self.netG, self.opt.lr,
                                              weight_decay=self.opt.weightDecay)
        
            # self.criterion = self.create_reg_criterion()
        self.n_tasks = len(self.opt.tasks)
        self.lr_sc = ReduceLROnPlateau(self.optimG, 'min', patience=500)
        
        if self.opt.display_id > 0:
            self.errors = OrderedDict()
            self.current_visuals = OrderedDict()
        if 'depth' in self.opt.tasks:
            self.criterion_reg = self.get_regression_criterion()
            
            ############AIF
        # if 'aif' in self.opt.tasks:
            # self.criterion_charbonnier = self.get_errors_charbonnier()
            
        if 'semantics' in self.opt.tasks:
            self.initialize_semantics()
        if 'instance' in self.opt.tasks:
            pass
        if 'normals' in self.opt.tasks:
            pass

    def initialize_semantics(self):
        from util.util import get_color_palette, get_dataset_semantic_weights
        self.global_cm = np.zeros((self.opt.n_classes-1, self.opt.n_classes-1))
        self.target = self.get_variable(torch.LongTensor(self.batchSize, self.opt.output_nc, self.opt.imageSize[0], self.opt.imageSize[1]))
        self.outG_np = None
        self.overall_acc = 0
        self.average_acc = 0
        self.average_iou = 0
        self.opt.color_palette = get_color_palette(self.opt.dataset_name)

        weights = self.get_variable(torch.FloatTensor(get_dataset_semantic_weights(self.opt.dataset_name)))
        self.cross_entropy = nn.CrossEntropyLoss(weight=weights)
    
    def train_batch(self, val_loader):
        self._train_batch(val_loader)

    def restart_variables(self):
        self.it = 0
        self.n_iterations = 0
        self.n_images = 0
        self.rmse = 0
        self.e_reg = 0
        self.norm_grad_sum = 0

    def mean_errors(self):
        if 'depth' in self.opt.tasks:
            rmse_epoch = self.rmse / self.n_images
            self.set_current_errors(RMSE=rmse_epoch)

    def get_errors_regression(self, target, output):
        if self.total_iter % self.opt.print_freq == 0:

            # gets valid pixels of output and target
            if not self.opt.no_mask:
                (output, target), n_valid_pixls = self.apply_valid_pixels_mask(output, target, value=self.opt.mask_thres)

            with torch.no_grad():
                e_regression = self.criterion_reg(output, target.detach())
                for k in range(output.shape[0]):
                    self.rmse += sqrt(self.mse_scaled_error(output[k], target[k], n_valid_pixls).item()) # mean through the batch
                    self.n_images += 1

            self.set_current_visuals(depth_gt=target.data,
                                        depth_out=output.data)
            self.set_current_errors(L1=e_regression.item())
            
            # return e_regression

    def get_errors_semantics(self, target, output, n_classes):
        # e_semantics = self.cross_entropy(output, target)
        if self.total_iter % self.opt.print_freq == 0:
            with torch.no_grad():
                target_sem_np = target.cpu().numpy()
                output_np = np.argmax(output.cpu().data.numpy(), axis=1)
                cm = confusion_matrix(target_sem_np.ravel(), output_np.ravel(), labels=list(range(n_classes)))
                self.global_cm += cm[1:,1:]

                # scores
                overall_acc = metrics.stats_overall_accuracy(self.global_cm)
                average_acc, _ = metrics.stats_accuracy_per_class(self.global_cm)
                average_iou, _ = metrics.stats_iou_per_class(self.global_cm)

                self.set_current_errors(OAcc=overall_acc, AAcc=average_acc, AIoU=average_iou)
                self.set_current_visuals(sem_gt=target.data[0].cpu().float().numpy(),
                                        sem_out=output_np[0])

            # return e_semantics
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def get_errors_instance(self, target, output):
        pass

    def get_errors_normals(self, target, output):
        pass

#####Smoothing loss#######
    def get_smooth_loss(disp, img):
                 """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
                 grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
                 grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

                 grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
                 grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

                 grad_disp_x *= torch.exp(-grad_img_x)
                 grad_disp_y *= torch.exp(-grad_img_y)
                 return grad_disp_x.mean() + grad_disp_y.mean()
##########################
    def _train_batch(self, val_loader):
        input_cpu, target_cpu, aif_cpu = next(self.data_iter)
        
        input_data = input_cpu.to(self.cuda)
        input_data.requires_grad = True
        self.set_current_visuals(input=input_data.data)
        batch_size = input_cpu.shape[0]
        
        aif_data = aif_cpu.to(self.cuda)

        outG, aif_pred = self.netG.forward(input_data)
        
        
        losses = []
        for i_task, task in enumerate(self.opt.tasks):
            target = target_cpu[i_task].to(self.cuda)


            #     self.get_errors_semantics(target, outG[i_task], n_classes=self.opt.outputs_nc[i_task])
            if task == 'depth':
                
                ####### VVVV Loss SSIM VVVV####
                # class SSIMLoss(SSIM):
                #     def forward(self, x, y):
                #         return 1. - super().forward(x, y)

                # criterion = SSIMLoss().cuda() #if you need GPU support
                # ssimloss = ssim(aif_pred, aif_data)
                ###Old
                ###New
                # print("min_data",torch.min(aif_data))
                # print("max_data",torch.max(aif_data))
                # print("min_pred",torch.min(aif_pred))
                # print("max_pred",torch.max(aif_pred))
                
                # ashdadshjd
#                Sloss = ssim(aif_data, aif_pred, data_range=-5)

                # loss = SSIMLoss(data_range=1.)
                # output: torch.Tensor = loss(x, y)
                # output.backward()
                
                
             
                # print ("ssim", Sloss)
                
                #######^^^^^ Loss SSIM ^^^^^^^####
                #####VVVVV Calculate AIF LOSS CHARBONNIER VVVV####
                charb_diff = torch.add(aif_pred, -aif_data)
                charb_error = torch.sqrt( charb_diff * charb_diff + 1e-6 )
                charb_elems = aif_data.shape[0] * aif_data.shape[1] * aif_data.shape[2] * aif_data.shape[3]  
                loss_aif = torch.sum(charb_error) / charb_elems
                # print ("chabr_loss", loss_aif)
                # total_loss = loss_aif + 3*(1-Sloss)
                # print("Total loss:",total_loss)
                #####For deblurring loss include this_>losses.append(loss_aif) 

                #####^^^^^^ Calculate AIF LOSS CHARBONNIER ^^^^####
                #######VVVV L 1 norm for deblurring ###VVVVV 
                l1 = nn.L1Loss()
                l1_loss=l1(aif_pred,aif_data)
                losses.append(0.1*l1_loss)
  
                #######^^^^ L 1 norm for deblurring ###^^^^^^

                # loss_aif = loss_chab + losss_ssim 
                
                # print('Aif_data',aif_data.size())
                # print('Pred',aif_pred.size())   
              
                # ssim_loss = 1-pytorch_ssim.ssim(aif_pred, aif_data).item()
                
               #  loss_aif = pytorch_ssim.SSIM()
                # print(loss_aif) +
                ##changeing the indexes
               
                
                # loss_aif_temp = loss_aif_charb + 0.01*loss_aif_ssim
                # loss_aif = loss_aif_charb 
                         
                # losses.append(0.01*loss_aif) 
                
                #losses.append(Add weight here*loss_aif)
                # save_pred = output_aif.view(output_aif.size(1), output_aif.size(2),output_aif.size(3)).data.cpu().numpy()
                # save_pred = np.transpose(save_pred, [1, 2, 0])
                # normalization_values = {'mean': np.array([0.485, 0.456, 0.406]),
                #         'std': np.array([0.229, 0.224, 0.225])}
                # save_pred = save_pred * normalization_values['std'] + normalization_values['mean']
                # save_pred = np.clip(save_pred, a_min=0.0, a_max=1.0)

                # save_gt = aif_img.view(aif_img.size(1), aif_img.size(2),aif_img.size(3)).data.cpu().numpy()
                # save_gt = np.transpose(save_gt, [1, 2, 0])
                # save_gt = save_gt * normalization_values['std'] + normalization_values['mean']
                # save_gt = np.clip(save_gt, a_min=0.0, a_max=1.0)

                # matplotlib.image.imsave('data/1.OurDecoder/Aif_ET/aif_'+str(j)+'.png', aif_pred)
                # matplotlib.image.imsave('data/1.OurDecoder/Aif_GT/GT_aif_'+str(j)+'.png', aif_data)
                
                # print(aif_data.size())
                # print(aif_pred.size())
                # self.get_errors_charbonnier(aif_data, aif_pred)
                
            # elif task == 'depth':
                ####VVVVVVAdd regularization hereVVVVVV####
                # import cv2
                output = torch.unsqueeze(outG[i_task][:,0,:,:], dim=1)
                depth = torch.unsqueeze(target, dim=1)
                # print(output.size())
                # print(depth.size())
               
                dt = depth-output
                kernely = torch.tensor([[[[1,1,1],[0,0,0],[-1,-1,-1]]]], dtype=torch.float).cuda()
                kernelx = torch.tensor([[[[1,0,-1],[1,0,-1],[1,0,-1]]]], dtype=torch.float).cuda()
                deltadt_x = torch.nn.functional.conv2d(dt, kernelx, bias=None, stride=1, padding=0, dilation=1, groups=1)
                deltadt_y = torch.nn.functional.conv2d(dt, kernely, bias=None, stride=1, padding=0, dilation=1, groups=1)
                deltadt_x = torch.abs(deltadt_x)
                deltadt_y = torch.abs(deltadt_y)
                gradient =torch.mean(deltadt_x  + deltadt_y)
                 #Smoothingness loss MonoDepth2VVVVVVV#########
                disp = outG[i_task][:,0,:,:]
                #mean_disp = disp.mean(2, True).mean(3, True)
                #norm_disp = disp / (mean_disp + 1e-7)
                #smooth_loss = get_smooth_loss(norm_disp, target)
                #loss_smooth= self.opt.disparity_smoothness * smooth_loss 

                #sloss=1- F.structural_similarity(target, outG[i_task][:,0,:,:], data_range=256, win_size=7, K=(0.01**2, 0.03**2))
                ############SSIM for depth### ssim(aif_data, aif_pred, data_range=-5)		 
                
                #sloss = l_ssim(target,outG[i_task][:,0,:,:])
               
                #sloss = torch.clamp((1 - F.SSIM(target,outG[i_task][:,0,:,:], data_range=1.0)) * 0.5, 0, 1)

                #print("target size", np.shape(target))
                #	 print("outG size", np.shape(outG[i_task][:,0,:,:]))
                #asddfasdasasdase
                #ssim( X, Y, data_range=255, size_average=False)
                #losses.append(0.001*(self.criterion_reg(target, outG[i_task][:,0,:,:]))+(gradient))
                #Lap_filter = torch.Tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
                #l11 = F.conv2d(losses, Lap_filter, padding=1)
                losses.append((self.criterion_reg(target, outG[i_task][:,0,:,:])))
                self.get_errors_regression(target, outG[i_task][:,0,:,:])
                #print("Depth Loss:", self.get_errors_regression
				             
            #print("losses size", np.shape(losses))
            print("losses", losses)
                # asdjsadvdamsv

            
            # TODO: ADD AIF ERROR
            # Calculate AIF LOSS CHARBONNIER
            # charb_diff = torch.add(output_aif, -aif_img)
            # charb_error = torch.sqrt( charb_diff * charb_diff + 1e-6 )
            # charb_elems = aif_img.shape[0] * aif_img.shape[1] * aif_img.shape[2] * aif_img.shape[3]  
            # loss_aif = torch.sum(charb_error) / charb_elems
         
            
        # 
        self.loss_error = sum(losses)
        # self.loss_error = (0.5*losses[0])+(0.5*losses[1])

        self.optimG.zero_grad()
        self.loss_error.backward()
        
        ###Scheduler
        self.optimG.step()
        # val_loss = self.calculate_val_loss(val_loader)
        # self.lr_sc.step(val_loss)
        #print(self.get_lr(self.optimG))

        self.n_iterations += 1 # outG[0].shape[0]

        with torch.no_grad():
            # show each loss
            for i, loss_task in enumerate(losses):
                self.set_current_errors_string('loss{}'.format(i), self.to_numpy(loss_task))
       

    def evaluate(self, data_loader, epoch):
        if self.opt.validate and self.total_iter % self.opt.val_freq == 0:
            self.get_eval_error(data_loader)
            self.visualizer.display_errors(self.val_errors, epoch, float(self.it)/self.len_data_loader, phase='val')
            message = self.visualizer.print_errors(self.val_errors, epoch, self.it, len(data_loader), 0)
            print('[Validation] ' + message)
            self.visualizer.display_images(self.val_current_visuals, epoch=epoch, phase='val')

    def calculate_val_loss(self, data_loader):
        # print('In Val loss fnc \n')
        model = self.netG.train(False)
        aif_pred_list = list()
        aif_data_list = list()
        losses = np.zeros(self.n_tasks)
        aif_err = 0.0
        with torch.no_grad():
            pbar_val = range(len(data_loader))
            data_iter = iter(data_loader)
            for _ in pbar_val:
                #pbar_val.set_description('[Validation]')
                input_cpu, target_cpu, aif_cpu = next(data_iter)#.next()
                input_data = input_cpu.to(self.cuda)
                aif_data = aif_cpu.to(self.cuda)
                
                outG, aif_pred = model.forward(input_data)
                aif_pred_list.append(aif_pred)
                aif_data_list.append(aif_data)
                
                for i_task, task in enumerate(self.opt.tasks):
                    target = target_cpu[i_task].to(self.cuda)
                    if task == 'semantics':
                        target_np = target_cpu[i_task].data.numpy()
                        output_np = np.argmax(outG[i_task].cpu().data.numpy(), axis=1)
                        cm = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(self.opt.outputs_nc[i_task])))
                        target_cpu[i_task] = target.data[0].cpu().float().numpy()
                        outG[i_task] = output_np[0]
                        loss, _ = metrics.stats_iou_per_class(cm[1:,1:])
                        losses[i_task] += loss
                    elif task == 'depth':
                        losses[i_task] += sqrt(nn.MSELoss()(target, outG[i_task]))
                        outG[i_task] = outG[i_task].data
                 
                
                        
                # TODO: ADD AIF ERROR
                ##L1
                l1 = nn.L1Loss()
                l1_loss=l1(aif_pred,aif_data)
                aif_err += l1_loss
                # total_loss = 0
                # for idx, value in enumerate(aif_pred_list):
                #     total_loss = total_loss + l1(aif_pred_list[idx], aif_data_list[idx])
                # aif_err += total_loss/len(aif_pred_list)
                ##Charb
                # charb_diff = torch.add(aif_pred, -aif_data)
                # charb_error = torch.sqrt( charb_diff * charb_diff + 1e-6 )
                # charb_elems = aif_data.shape[0] * aif_data.shape[1] * aif_data.shape[2] * aif_data.shape[3]  
                # loss_aif = torch.sum(charb_error) / charb_elems
                # aif_err += loss_aif
                # losses.append(self.criterion_reg(aif_data, aif_pred))
                # loss_aif =  self.get_errors_regression(aif_data, aif_pred)
                # aif_err += loss_aif
                return aif_err
        
    def get_eval_error(self, data_loader):
        model = self.netG.train(False)
        self.val_errors = OrderedDict()
        self.val_current_visuals = OrderedDict()
        aif_pred_list = list()
        aif_data_list = list()
        losses = np.zeros(self.n_tasks)
        aif_err = 0.0
        with torch.no_grad():
            pbar_val = tqdm(range(len(data_loader)))
            data_iter = iter(data_loader)
            for _ in pbar_val:
                pbar_val.set_description('[Validation]')
                input_cpu, target_cpu, aif_cpu = next(data_iter)#.next()
                input_data = input_cpu.to(self.cuda)
                aif_data = aif_cpu.to(self.cuda)
                
                outG, aif_pred = model.forward(input_data)
                aif_data_list.append(aif_data)
                aif_pred_list.append(aif_pred)
                
                for i_task, task in enumerate(self.opt.tasks):
                    target = target_cpu[i_task].to(self.cuda)
                    if task == 'semantics':
                        target_np = target_cpu[i_task].data.numpy()
                        output_np = np.argmax(outG[i_task].cpu().data.numpy(), axis=1)
                        cm = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(self.opt.outputs_nc[i_task])))
                        target_cpu[i_task] = target.data[0].cpu().float().numpy()
                        outG[i_task] = output_np[0]
                        loss, _ = metrics.stats_iou_per_class(cm[1:,1:])
                        losses[i_task] += loss
                    elif task == 'depth':
                        losses[i_task] += sqrt(nn.MSELoss()(target, outG[i_task]))
                        outG[i_task] = outG[i_task].data
                        # print("outg",np.shape( outG[i_task]))
                        
                 
                
                        
                # TODO: ADD AIF ERROR
                #L1
                l1 = nn.L1Loss()
                l1_loss=l1(aif_pred,aif_data)
                aif_err += l1_loss
                # total_loss = 0
                # for idx, value in enumerate(aif_pred_list):
                #     total_loss = total_loss + l1(aif_pred_list[idx], aif_data_list[idx])
                # aif_err += total_loss/len(aif_pred_list)
                #charb
                
                ##VVVCharbVVV##
                # charb_diff = torch.add(aif_pred, -aif_data)
                # charb_error = torch.sqrt( charb_diff * charb_diff + 1e-6 )
                # charb_elems = aif_data.shape[0] * aif_data.shape[1] * aif_data.shape[2] * aif_data.shape[3]  
                # loss_aif = torch.sum(charb_error) / charb_elems
                # aif_err += loss_aif
                ####END^^^^###
                # print("aif_err",np.shape(aif_err))
                # adjhsadjs
                

            self.val_current_visuals.update([('input', input_cpu)])
            for i_task, task in enumerate(self.opt.tasks):
                self.val_errors.update([('l_{}'.format(task), losses[i_task]/len(data_loader))])
    
                self.val_current_visuals.update([('t_{}'.format(task), target_cpu[i_task])])
                self.val_current_visuals.update([('o_{}'.format(task), outG[i_task])])
                
            self.val_errors.update([('a_AIF', aif_err.cpu().data.numpy())])
            # self.val_errors.update([('a_AIF', aif_err)])

    def set_current_errors_string(self, key, value):
        self.errors.update([(key, value)])

    def set_current_errors(self, **k_dict_elements):
        for key, value in k_dict_elements.items():
            self.errors.update([(key, value)])

    def get_current_errors(self):
        return self.errors

    def get_current_errors_display(self):
        return self.errors

    def set_current_visuals(self, **k_dict_elements):
        for key, value in k_dict_elements.items():
            self.current_visuals.update([(key, value)])

    def get_current_visuals(self):
        return self.current_visuals

    def get_checkpoint(self, epoch):
        return ({'epoch': epoch,
                 'arch_netG': self.opt.net_architecture,
                 'state_dictG': self.netG.state_dict(),
                 'optimizerG': self.optimG,
                 'best_pred': self.best_val_error,
                 'tasks': self.opt.tasks,
                 'mtl_method': self.opt.mtl_method,
                #  'data_augmentation': self.opt.data_augmentation, # used before loading net
                 'n_classes': self.opt.n_classes,
                 })

    def load_network(self):
        if self.opt.epoch is not 'latest' or self.opt.epoch is not 'best':
            self.opt.epoch = self.opt.epoch.zfill(4)
        checkpoint_file = os.path.join(self.checkpoints_path, self.opt.epoch + '.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
            self.start_epoch = checkpoint['epoch']
            self.opt.net_architecture = checkpoint['arch_netG']
            self.opt.n_classes = checkpoint['n_classes']
            self.opt.mtl_method = checkpoint['mtl_method']
            self.opt.tasks = checkpoint['tasks']
            netG = self.create_network()
            netG.load_state_dict(checkpoint['state_dictG'])
            optimG = checkpoint['optimizerG']
            self.best_val_error = checkpoint['best_pred']
            self.print_save_options()
            print("Loaded model from epoch {}".format(self.start_epoch))
            return netG, optimG
        else:
            raise ValueError("Couldn't find checkpoint on path: {}".format(self.checkpoints_path + '/' + self.opt.epoch))

    def load_weights_from_pretrained_model(self):
        epoch = 'best'
        checkpoint_file = os.path.join(self.opt.pretrained_path, epoch + '.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print("Loading {} checkpoint of model {} ...".format(epoch, self.opt.pretrained_path))
            self.opt.net_architecture = checkpoint['arch_netG']
            self.opt.n_classes = checkpoint['n_classes']
            self.opt.mtl_method = checkpoint['mtl_method']
            self.opt.tasks = checkpoint['tasks']
            netG = self.create_network()
            model_dict = netG.state_dict()
            pretrained_dict = checkpoint['state_dictG']
            model_shapes = [v.shape for k, v in model_dict.items()]
            exclude_model_dict = [k for k, v in pretrained_dict.items() if v.shape not in model_shapes]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in exclude_model_dict}
            model_dict.update(pretrained_dict)
            netG.load_state_dict(model_dict)
            _epoch = checkpoint['epoch']
            # netG.load_state_dict(checkpoint['state_dictG'])
            print("Loaded model from epoch {}".format(_epoch))
            return netG
        else:
            raise ValueError("Couldn't find checkpoint on path: {}".format(self.pretrained_path + '/' + epoch))

    def to_numpy(self, data):
        return data.data.cpu().numpy()