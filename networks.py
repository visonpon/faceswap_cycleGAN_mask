import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import option as opt

class Generator(nn.Module):
    def __init__(self, opt.input_nc, opt.output_nc, opt.ngf, norm_layer=nn.BatchNorm2d, opt.use_dropout, opt.n_blocks, opt.gpu_ids, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.ngf = opt.ngf
        self.gpu_ids = opt.gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(opt.input_nc, opt.ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            output= nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        mask = F.sigmoid(output[:, :1])
        oimg = output[:, 1:]
        mask = mask.repeat(1, 3, 1, 1)
        oimg = oimg*mask + input*(1-mask)

        return oimg, mask
   
  
  
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Discriminator(nn.Module):
    def __init__(self,gpu_ids=[]):
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.conv_block = nn.Sequential(
            nn.Conv2d(3,   64,  4, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,  128,  4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,  256,  4, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256,  512, 4, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 4),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 1, 1),
        )

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            output =  nn.parallel.data_parallel(self.conv_block, input, self.gpu_ids)
        else:
            output = self.conv_block
            
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
