import networks
import option as opt
from PIL import Image
import torch
from torch.optim import lr_scheduler
def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.netG_A = networks.Generator(opt.input_nc, opt.output_nc,
                                                opt.ngf, opt.norm, not opt.no_dropout,opt.gpu_ids)
        self.netG_B = networks.Generator(opt.input_nc, opt.output_nc,
                                                opt.ngf, opt.norm, not opt.no_dropout,opt.gpu_ids)

        self.netD_A = networks.Discriminator().cuda()
        self.netD_B = networks.Discriminator().cuda()
        
        #self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.criterionGAN = nn.MSELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.real_label = 1.0
        self.fake_label = 0.0
        self.real_tensor = self.Tensor(input.size()).fill_(self.real_label)
        self.real_label_var = Variable(self.real_tensor, requires_grad=False)
        self.target_tensor = self.real_label_var
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam( list(self.netG_A.parameters())+list(self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D_A)
        self.optimizers.append(self.optimizer_D_B)
        for optimizer in self.optimizers:
            self.schedulers.append(lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1))
        
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def set_input(self,input):
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.A_img = Image.open(self.A_path).convert('RGB')
        self.B_img = Image.open(self.B_path).convert('RGB')
        self.A = transform(A_img)
        self.B = transform(B_img)

        
def forward(self):
        self.real_A = Variable(self.A) ##############
        self.real_B = Variable(self.B)
        
        
def backward_D_basic(self, netD, real, fake):
    # Real
    pred_real = netD(real)
    loss_D_real = self.criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = self.criterionGAN(pred_fake, False)
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    # backward
    loss_D.backward()
    return loss_D

def backward_D_A(self):
    fake_B = self.fake_B_pool.query(self.fake_B)
    loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
    self.loss_D_A = loss_D_A.data[0]

def backward_D_B(self):
    fake_A = self.fake_A_pool.query(self.fake_A)
    loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    self.loss_D_B = loss_D_B.data[0]

def backward_G(self):
    lambda_idt = self.opt.identity
    lambda_A = self.opt.lambda_A
    lambda_B = self.opt.lambda_B
    # Identity loss
    if lambda_idt > 0:
        # G_A should be identity if real_B is fed.
        idt,_ = self.netG_A(self.real_B)
        loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
        # G_B should be identity if real_A is fed.
        idt_B,_ = self.netG_B(self.real_A)
        loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

        self.idt_A = idt_A.data
        self.idt_B = idt_B.data
        self.loss_idt_A = loss_idt_A.data[0]
        self.loss_idt_B = loss_idt_B.data[0]
    else:
        loss_idt_A = 0
        loss_idt_B = 0
        self.loss_idt_A = 0
        self.loss_idt_B = 0

    # GAN loss D_A(G_A(A))
    fake_B,mask_B = self.netG_A(self.real_A)
    pred_fake = self.netD_A(fake_B)
    loss_G_A = self.criterionGAN(pred_fake, self.target_tensor)#################
    loss_G_A += torch.mean(torch.abs(mask_B))

    # GAN loss D_B(G_B(B))
    fake_A,mask_A = self.netG_B(self.real_B)
    pred_fake = self.netD_B(fake_A)
    loss_G_B = self.criterionGAN(pred_fake, self.target_tensor)
    loss_G_B += torch.mean(torch.abs(mask_A))

    # Forward cycle loss
    rec_A,_ = self.netG_B(fake_B)
    loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

    # Backward cycle loss
    rec_B,_ = self.netG_A(fake_A)
    loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B
    # combined loss
    loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
    loss_G.backward()

    self.fake_B = fake_B.data
    self.fake_A = fake_A.data
    self.rec_A = rec_A.data
    self.rec_B = rec_B.data

    self.loss_G_A = loss_G_A.data[0]
    self.loss_G_B = loss_G_B.data[0]
    self.loss_cycle_A = loss_cycle_A.data[0]
    self.loss_cycle_B = loss_cycle_B.data[0]

def optimize_parameters(self):
    # forward
    self.forward()
    # G_A and G_B
    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()
    # D_A
    self.optimizer_D_A.zero_grad()
    self.backward_D_A()
    self.optimizer_D_A.step()
    # D_B
    self.optimizer_D_B.zero_grad()
    self.backward_D_B()
    self.optimizer_D_B.step()
   
def save(self, label):
    self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
    self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
    self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
    self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        
        
def update_learning_rate(self):
    for scheduler in self.schedulers:
        scheduler.step()
    lr = self.optimizers[0].param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
