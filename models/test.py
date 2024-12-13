import os
import shutil
from tqdm import tqdm
import time
from collections import OrderedDict
from ipdb import set_trace as st
import random
import numpy as np
from PIL import Image

import re

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn

#from util.visualizer import Visualize
import networks as networks


#from dataloader.data_loader import CreateDataLoader

net_architecture =''
def create_G_network():
    netG = networks.define_G(3, 1, 64, net_architecture=net_architecture, gpu_ids='')
    # print(netG)
    return netG



def load_network():
    cuda = True
    checkpoint_file = 'E:\\D3NET\\pytorch\\checkpoints\\Train Model\\latest.pth.tar'
    if os.path.isfile(checkpoint_file):
        #print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        net_architecture = checkpoint['arch_netG']
        netG = create_G_network()
        try:
            n_classes = checkpoint['n_classes']
            mtl_method = checkpoint['mtl_method']
            tasks = checkpoint['tasks']
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
        netG.load_state_dict(pretrained_dict)
        if cuda:
            cuda = torch.device('cuda:0') # set externally. ToDo: set internally
            netG = netG.cuda()
        best_val_error = checkpoint['best_pred']

        print("Loaded model from epoch {}")
        return netG
    else:
        print("Couldn't find checkpoint on path: {}")

network = load_network()
