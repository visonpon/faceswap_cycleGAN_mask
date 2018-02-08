import networks
import option as opt
from torchvision.utils import save_image
import itertools
import torchvision
import torch
#####################
real = ""
n_epochs = 60
batch_size = 1
#####################


transformer = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
            lambda img: img[:, 1:-1, 1:-1]),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
dataset = torchvision.datasets.ImageFolder('data/', transformer)

labels_neg = [i for i, (_, l) in enumerate(dataset.imgs) if l == 0]
labels_pos = [i for i, (_, l) in enumerate(dataset.imgs) if l == 1]

sampler_neg = torch.utils.data.sampler.SubsetRandomSampler(labels_neg)
sampler_pos = torch.utils.data.sampler.SubsetRandomSampler(labels_pos)

pos_loader = torch.utils.data.DataLoader(dataset,sampler=sampler_pos,batch_size=batch_size,)

neg_loader = torch.utils.data.DataLoader(dataset,sampler=sampler_neg,batch_size=batch_sise,)

def mask(input):
    netG = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout,opt.gpu_ids)
    _,mask = netG(input)
    return mask

for epoch in xrange(n_epochs):
    for (pos, _), (neg, _) in itertools.izip(pos_loader, neg_loader):
        mask_pos = mask(pos)
        mask_neg = mask(neg)
        save_image(mask_pos.data.cpu()[0],'mask_pos.png')
        save_image(mask_neg.data.cpu()[0],'mask_neg.png') #################
