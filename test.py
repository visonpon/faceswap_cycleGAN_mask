from networks import *
netG_A = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout,opt.gpu_ids)
netG_B = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout,opt.gpu_ids)
netD_A = networks.Discriminator().cuda()
netD_B = networks.Discriminator().cuda()

save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
       
try:
    netG_A.load_state_dict(torch.load(os.path.join(self.save_dir, save_filename1)))
    netG_B.load_state_dict(torch.load(os.path.join(self.save_dir, save_filename2)))
    netD_A.load_state_dict(torch.load(os.path.join(self.save_dir, save_filename3)))
    netD_A.load_state_dict(torch.load(os.path.join(self.save_dir, save_filename4)))
    print ("model loaded.")
except:
    print ("Weights file not found.")
    pass 
    
def cycle_variables(netG):
    input = netG.inputs[0]
    fake_output = netG.outputs[0]
    mask_output = netG.ouputs[1]
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
    rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
    
    masked_fake_output = alpha * rgb + (1-alpha) * input 

    fn_generate = K.function([distorted_input], [masked_fake_output])
    fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
    fn_abgr = K.function([distorted_input], [concatenate([alpha, rgb])])
    return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr
    
"""
path_A: A function that takes distorted_A as input and outputs fake_A.
path_B: A function that takes distorted_B as input and outputs fake_B.
path_mask_A: A function that takes distorted_A as input and outputs mask_A.
path_mask_B: A function that takes distorted_B as input and outputs mask_B.
path_abgr_A: A function that takes distorted_A as input and outputs concat([mask_A, fake_A]).
path_abgr_B: A function that takes distorted_B as input and outputs concat([mask_B, fake_B]).
real_A: A (batch_size, 64, 64, 3) tensor, target images for generator_A given input distorted_A.
real_B: A (batch_size, 64, 64, 3) tensor, target images for generator_B given input distorted_B.
"""
