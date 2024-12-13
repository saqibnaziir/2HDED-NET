import os
import shutil
from tqdm import tqdm
import time
from collections import OrderedDict
from ipdb import set_trace as st
import random
import numpy as np
from PIL import Image
import matplotlib.image
import matplotlib.pyplot as plt

import re

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn

from util.visualizer import Visualizer
import networks.networks as networks


from dataloader.data_loader import CreateDataLoader

class MTL_Test():
    def name(self):
        return 'Test Model for MTL'

    def initialize(self, opt):
        self.opt = opt
        self.opt.imageSize = self.opt.imageSize if len(self.opt.imageSize) == 2 else self.opt.imageSize * 2
        self.gpu_ids = ''
        self.batchSize = self.opt.batchSize
        self.checkpoints_path = os.path.join(self.opt.checkpoints, self.opt.name)
        self.create_save_folders()

        self.netG = self.load_network()

        self.data_loader, _ = CreateDataLoader(opt)

        # visualizer
        self.visualizer = Visualizer(self.opt)
        if 'semantics' in self.opt.tasks:
            from util.util import get_color_palette
            self.opt.color_palette = np.array(get_color_palette(self.opt.dataset_name))
            self.opt.color_palette = list(self.opt.color_palette.reshape(-1))

    def load_network(self):
        if self.opt.epoch is not 'latest' or self.opt.epoch is not 'best':
            self.opt.epoch = self.opt.epoch.zfill(4)
        checkpoint_file = os.path.join(self.checkpoints_path, self.opt.epoch + '.pth.tar')
        print(checkpoint_file)
        if os.path.isfile(checkpoint_file):
            print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
            checkpoint = torch.load(checkpoint_file)
            self.start_epoch = checkpoint['epoch']
            self.opt.net_architecture = checkpoint['arch_netG']
            netG = self.create_G_network()
            try:
                self.opt.n_classes = checkpoint['n_classes']
                self.opt.mtl_method = checkpoint['mtl_method']
                self.opt.tasks = checkpoint['tasks']
            except:
                pass
            pretrained_dict = checkpoint['state_dictG']
            pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(pretrained_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    pretrained_dict[new_key] = pretrained_dict[key]
                    del pretrained_dict[key]
            netG.load_state_dict(pretrained_dict, strict=False)
            if self.opt.cuda:
                self.cuda = torch.device('cuda:0') # set externally. ToDo: set internally
                netG = netG.cuda()
            self.best_val_error = checkpoint['best_pred']

            print("Loaded model from epoch {}".format(self.start_epoch))
            return netG
        else:
            print("Couldn't find checkpoint on path: {}".format(self.checkpoints_path + '/' + self.opt.epoch))


    def test(self):
        print('Test phase using {} split.'.format(self.opt.test_split))
        data_iter = iter(self.data_loader)
        self.netG.eval()
        total_iter = 0

        for it in tqdm(range(len(self.data_loader))):
            total_iter += 1
            input_cpu, targets_cpu, aif_data = next(data_iter)#.next()
            # print("input_cpu", type(input_cpu))
            # print("target_cpu", type(targets_cpu))
            # print("aif_data", type(aif_data))
            ######Converting list into tensor torch
            target_temp=torch.stack(targets_cpu)
            # print("target_cpu2", type(target_temp))
            # asdasdasdas
            #Save NP.Arry of Input_CPU GT VVVVV####
            # depth_input_temp = (target_temp.unsqueeze(0).cuda()).detach().cpu().clone().numpy()
            np.save('Results/gt_depth/target_cpu'+str(total_iter)+'.npy', target_temp)
            ###########^^^^^^^^^^^############
            
            #Save NP.Arry of AIF GT VVVVV####
            aif_data_temp = (aif_data.unsqueeze(0).cuda()).detach().cpu().clone().numpy()
            np.save('Results/aif_data/aif_gt'+str(total_iter)+'.npy', aif_data_temp)
            ###########^^^^^^^^^^^############
            
            # self.save_as_png(tensor = aif_data, filename="aif_data/aif_data" + str(total_iter) + ".png")
            # self.save_as_png(tensor = aif_data, filename="aif_data/aif_data" + str(total_iter) + ".png")
            # print(input_cpu.size())
            # print(targets_cpu.size())
            # print(aif_pred.size())
            
            input_gpu = input_cpu.to(self.cuda)

            with torch.no_grad():
                outG_gpu, aif_pred = self.netG.forward(input_gpu)
                # print("out", type(outG_gpu))
                # print("aif", type(aif_pred))
                
                # print("outG_gpu", type(outG_gpu))
                # print("target_cpu", type(targets_cpu))
                # print("aif_data", type(aif_data))
                ######Converting list into tensor torch
                outG_gpu_temp=torch.stack(outG_gpu)
                # print("outG_gpu_temp2", type(outG_gpu_temp))
                
                #Save NP.Arry of outG_gpu GT VVVVV####
                depth_input_temp = (outG_gpu_temp.unsqueeze(0).cuda()).detach().cpu().clone().numpy()
                np.save('Results/pred_depth/outG_gpu'+str(total_iter)+'.npy', depth_input_temp)
                # asdasdasdas
                
                
                # print("tempaif", type(aif_temp))
                # aif_temp = aif_pred.squeeze()
                # print(type(np.asarray(outG_gpu[0][0])))
               
                #Save NP.Arry of AIF PredictionsVVVVVVVVVVV
                aif_temp = (aif_pred.unsqueeze(0).cuda()).detach().cpu().clone().numpy()
                # print("air_temp",type(aif_temp))
                np.save('Results/aif_pred/aif_pred'+str(total_iter)+'.npy', aif_temp)
                # print(np.shape(np.asarray(outG_gpu[0][0])))
                # print("air_temp",np.shape(aif_temp))
                ######################^^^^^^^^^^^
                # matplotlib.image.imsave('aif_pred/aif_pred'+str(total_iter)+'.png', aif_temp)
                # print(np.shape(outG_gpu[0][0]))   
                # self.save_as_png(tensor = torch.tensor(outG_gpu), filename="aif_pred/aif_pred" + str(total_iter) + ".png")
            
            if self.opt.save_samples:
                self.save_images(input_gpu, outG_gpu, targets_cpu, it + 1, 'test')

    def create_G_network(self):
        netG = networks.define_G(self.opt.input_nc, self.opt.output_nc, 64, net_architecture=self.opt.net_architecture, opt=self.opt, gpu_ids='')
        # print(netG)
        return netG

    def save_depth_as_png(self, data, filename):
        """
        All depths are saved in np.uint16
        """
        data_np = data.data[0].cpu().float().numpy()
        data_np = data_np * self.opt.scale_to_mm
        data_np = data_np.astype(np.uint16)
        data_pil = Image.fromarray(np.squeeze(data_np), mode='I;16').convert(mode='I')

        data_pil.save(filename)

    # def save_semantics_as_png(self, data, filename):
    #     from util.util import labels_to_colors
    #     # data_pil = Image.fromarray(np.squeeze(labels_to_colors(data, self.opt.color_palette).astype(np.)))
    #     # data_tensor = 
    #     data = data.cpu().data[0].numpy()
    #     if 'output' in filename:
    #         data = np.argmax(data, axis=0)
    #     data = data.astype(np.uint8)
    #     data_pil = Image.fromarray(data).convert('P')
    #     data_pil.putpalette(self.opt.color_palette)

        # data_pil.save(filename)

    def save_rgb_as_png(self, data, filename):
        data_np = data.data[0].cpu().float().numpy()
        data_np = np.transpose(data_np, (1,2,0))
        data_np = ((data_np + 1) / 2) * 255
        data_np = data_np.astype(np.uint8)
        data_pil = Image.fromarray(np.squeeze(data_np), mode='RGB')

        data_pil.save(filename)

    def save_as_png(self, tensor, filename):
        if 'depth' in filename:
            self.save_depth_as_png(data=tensor, filename=filename)
        # elif 'semantic' in filename:
        #     self.save_semantics_as_png(data=tensor, filename=filename)
        elif 'input' in filename:
            self.save_rgb_as_png(data=tensor, filename=filename)
        # else:
        #     self.save_rgb_as_png(data=tensor, filename=filename)

    def create_save_folders(self, subfolders=['input', 'target', 'output', 'input_aif', 'output_aif']):
        if self.opt.save_samples:
            if self.opt.test:
                self.save_samples_path = os.path.join('results', self.opt.model, self.opt.name, self.opt.epoch)
                for subfolder in subfolders:
                    path = os.path.join(self.save_samples_path, subfolder)
                    os.system('mkdir -p {0}'.format(path))
                    if 'input' not in subfolder:
                        for task in self.opt.tasks:
                            path = os.path.join(self.save_samples_path, subfolder, task)
                            os.system('mkdir -p {0}'.format(path))

    def save_images(self, input, outputs, targets, index, phase='train'):
        # save other images
        self.save_as_png(input.data, '{}/input/input_{:04}.png'.format(self.save_samples_path, index))
        for i, target in enumerate(targets):
            self.save_as_png(outputs[i], '{}/output/{}/output_{}_{:04}.png'.format(self.save_samples_path, self.opt.tasks[i], self.opt.tasks[i], index))
            self.save_as_png(target, '{}/target/{}/target_{}_{:04}.png'.format(self.save_samples_path, self.opt.tasks[i], self.opt.tasks[i], index))
