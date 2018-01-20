import torch
import torch.nn as nn
import torchvision

from mnetworks import *

#init some hyperparameter

#init training data

#init model
self.netG_A = networks.Generator(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.norm, not opt.no_dropout,opt.gpu_ids)
self.netG_B = networks.Generator(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.norm, not opt.no_dropout,opt.gpu_ids)

self.netD_A = networks.Discriminator
# training


#save model
