import torch
import torch.nn as nn
import torchvision
import option as opt
from networks import *
import model
import data_process
#init some hyperparameter

#init training data
data_loader = CustomDatasetDataLoader()
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

#init model
self.netG_A = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.no_dropout, opt.gpu_ids)
self.netG_B = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.no_dropout, opt.gpu_ids)

self.netD_A = networks.Discriminator().cuda()
self.netD_B = networks.Discriminator().cuda()

print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        networks.print_network(self.netD_A)
        networks.print_network(self.netD_B)
print('-----------------------------------------------')

# training
print('Training...')
for epoch in range(0, opt.niter+ 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            t = (time.time() - iter_start_time) / opt.batchSize
            
            print('Epoch #%d' % (epoch+1))
            print('Batch #%d' % opt.batchSize)
            print('Loss G_A: %0.3f' % loss_G_A.data[0] + '\t' +
                  'Loss G_B: %0.3f' % loss_G_A.data[0])
            print('Loss D_A: %0.3f' % loss_D_A.data[0] + '\t' +
                  'Loss D_B: %0.3f' % loss_D_A.data[0])
            print('Loss D_A: %0.3f' % loss_D_A.data[0] + '\t' +
                  'Loss D_B: %0.3f' % loss_D_A.data[0])
            print('-'*50)
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

#save model
