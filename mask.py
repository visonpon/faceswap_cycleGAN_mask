import networks
import option as opt

def mask(input):
    netG = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout,opt.gpu_ids)
    _,mask = netG(input)
    return mask
        
