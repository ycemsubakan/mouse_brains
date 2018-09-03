import numpy as np
import torch
from algorithms_v2 import netD, netG, adversarial_trainer, VAE, NF_changedim, compute_nparam_density, conv_autoenc_mice, netg_dcgan, netd_dcgan, conv_VAE_mouse, conv_VAE_mouse_v3, adversarial_wasserstein_trainer, netd_dcgan_par, netg_dcgan_par, weights_init_seq, get_scores, compute_mmd
import pdb
from torch.autograd import Variable
import matplotlib.pyplot as plt
import utils as ut
import os
from drawnow import drawnow, figure
import visdom 
import torch.optim.lr_scheduler as tol
import torch.nn.functional as F
import itertools as it
import torch.nn as nn
import sklearn.mixture as mix
import sklearn.manifold as mnf
import pickle
from torchvision import datasets, transforms
import argparse
from WassersteinGAN.models.dcgan import DCGAN_G
import nrrd
import skimage.feature as skf
import socket
import time
import skimage.draw as skd

vis = visdom.Visdom(port=0, server='http://yourserver', env='your_env2')
assert vis.check_connection()

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='choose your model ', default='NF')
argparser.add_argument('--dopca', type=int, default=0)
arguments = argparser.parse_args()

np.random.seed(2)
torch.manual_seed(9)
arguments.cuda = torch.cuda.is_available()
arguments.batch_size = 128
arguments.data = 'mice_nonregistered'
arguments.input_type = 'autoenc'

home = os.path.expanduser('~')
hostname = socket.gethostname()

if hostname == 'nmf':
    train_data_dir = '/your_root/data2/unregistered_images/'
    test_data_dir = '/your_root/data2/unregistered_images_test/'
elif hostname == 'cem-gpu':
    train_data_dir = '/home/cem/unregistered_images/'
    test_data_dir = '/home/cem/unregistered_images_test/'

transform = transforms.Compose([
        #transforms.Grayscale(),
        transforms.Resize(size=(320, 456)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
dset_train = datasets.ImageFolder(train_data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=24, shuffle=True,
                                           pin_memory=True, num_workers=arguments.num_gpus)

dset_test = datasets.ImageFolder(test_data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=24, shuffle=False,
                                          pin_memory=True, num_workers=arguments.num_gpus)


# load the model
Ks = [200, 200]
    
mdl = conv_VAE_mouse_v3(320, 456, Ks, M=64, num_gpus=arguments.num_gpus) 
   
if arguments.cuda:
    mdl.cuda()

#model_desc = 'VAE_arc3_l1_lrelu_d48'
model_desc = 'VAE_arc5_l1'

path = 'models/' + model_desc + '_{}_K_{}.t'.format(arguments.data, Ks)
if 1 & os.path.exists(path):
    mdl.load_state_dict(torch.load(path))

nbatches = 75 
hhat_path = 'mouse_embeddings/' + model_desc + '_hats.t'
if 1 & os.path.exists(hhat_path):
    #all_xs = []
    #if arguments.dopca:
    #    for i, (data, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
    #        if arguments.cuda:
    #            data = data.cuda()

    #        all_xs.append(data.squeeze())
    #        print(i)

    #    all_xs_cat = torch.cat(all_xs, dim=0)

    dcts = torch.load(hhat_path)
    all_hhats_cat = dcts['hhats']
    all_xs_cat = dcts['xs']

else:
    all_hhats = []
    all_xhats = []
    all_xs = []
    for i, (data, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
        if arguments.cuda:
            data = data.cuda()

        if 1:
            data = data[:, :2, :, :]
    
        hhat = nn.parallel.data_parallel(mdl.encoder, Variable(data), range(arguments.num_gpus))
        xhat = nn.parallel.data_parallel(mdl.decoder, hhat[:, :Ks[0]], range(arguments.num_gpus))
        all_hhats.append(hhat.data.squeeze())
        all_xhats.append(xhat.data.squeeze())
        all_xs.append(data.squeeze())
        print(i)
    all_hhats_cat = torch.cat(all_hhats, dim=0)[:, :Ks[0]]
    all_xs_cat = torch.cat(all_xs, dim=0)

    dct = {'hhats' : all_hhats_cat, 
           'xs' : all_xs_cat}
    if not os.path.exists('mouse_embeddings'):
        os.mkdir('mouse_embeddings')
    torch.save(dct, hhat_path)

folder = 'unregistered_missingdata_results'
##### do pca see what happens

all_errs = []
ntype = 'circle'
if 1:
    mean_x = all_xs_cat.mean(0, keepdim=True)

    all_xs_cat_c = (all_xs_cat - mean_x).view(all_xs_cat.size(0), -1)

    Ureal, Sreal, _ = torch.svd(all_xs_cat_c.t().cpu())

    W = Ureal[:, :200].cuda()

    fls = torch.load(folder + '/pca_hhat.t')
    fls_small = torch.load(folder + '/pca_hhat_rad40.t')

    masks = Variable(fls[1].cuda())
    masks_small = Variable(fls_small[1].cuda())

    all_tdata = []
    for tdata, _ in test_loader:
        if 1:
            tdata = tdata[:, :2, :, :]
        all_tdata.append(tdata)
    all_tdata_cat = Variable(torch.cat(all_tdata, 0).cuda())

    hhat = Variable(fls[0], requires_grad=True)
    hhat_small = Variable(fls_small[0])

    recons_pca = torch.matmul(Variable(W), hhat.t()).t() + Variable(mean_x.view(1, -1).cuda())
    recons_pca = recons_pca.data
    recons_pca[recons_pca>1] = 1
    recons_pca[recons_pca<-1] = -1 
   
    recons_pca = recons_pca.view(-1, 2, 320, 456)
    recons_pca = torch.cat([recons_pca, -torch.ones(recons_pca.size(0), 1, 320, 456).cuda()], dim=1)


    recons_pca_small = torch.matmul(Variable(W), hhat.t()).t() + Variable(mean_x.view(1, -1).cuda())
    recons_pca_small = recons_pca_small.data
    recons_pca_small[recons_pca_small>1] = 1
    recons_pca_small[recons_pca_small<-1] = -1 
   
    recons_pca_small = recons_pca_small.view(-1, 2, 320, 456)
    recons_pca_small = torch.cat([recons_pca_small, -torch.ones(recons_pca_small.size(0), 1, 320, 456).cuda()], dim=1)


if 1:
        
    # get the masks right
    fls = torch.load(folder + '/conv_net_hhat.t')
    fls_small = torch.load(folder + '/conv_net_hhat_rad40.t')

    masks_conv = (fls[1].cuda())
    masks_conv = torch.cat([masks_conv.data.cpu(), torch.ones(masks_conv.size(0), 1, 320, 456)], dim=1)

    masks_conv_small = (fls_small[1].cuda())
    masks_conv_small = torch.cat([masks_conv_small.data.cpu(), torch.ones(masks_conv_small.size(0), 1, 320, 456)], dim=1)

    all_tdata = []
    for tdata, _ in test_loader:
        if 1: 
            tdata = tdata[:, :2, :, :]
        all_tdata.append(tdata)
    all_tdata_cat = Variable(torch.cat(all_tdata, 0).cuda())

    hhat = Variable(fls[0], requires_grad=True)
    hhat_small = Variable(fls_small[0])

    recons_conv = nn.parallel.data_parallel(mdl.decoder, (hhat.unsqueeze(-1).unsqueeze(-1)) , range(arguments.num_gpus)).data.cpu()

    recons_conv = torch.cat([recons_conv, -torch.ones(recons_conv.size(0), 1, 320, 456)], dim=1)

    recons_conv_small = nn.parallel.data_parallel(mdl.decoder, (hhat_small.unsqueeze(-1).unsqueeze(-1)) , range(arguments.num_gpus)).data.cpu()

    recons_conv_small = torch.cat([recons_conv_small, -torch.ones(recons_conv_small.size(0), 1, 320, 456)], dim=1)



    #score = (((recons_conv - all_tdata_cat)*(1 - masks.float())).abs()).mean()

Nims = 6
ncols = 6

all_tdata_cat = torch.cat([all_tdata_cat, -torch.ones(all_tdata_cat.size(0), 1, 320, 456).cuda()], dim=1).data.cpu()
real_data = ut.collate_images_rectangular_color(all_tdata_cat, N=Nims, ncols=ncols, L1=320, L2=456)
all_tdata_cat = all_tdata_cat[:80]

masks = torch.cat([masks.data, (torch.ones(masks.size(0), 1, 320, 456).cuda())], dim=1).cpu()
masked_data = ut.collate_images_rectangular_color(all_tdata_cat.data * masks, N=Nims, ncols=ncols, L1=320, L2=456)
masked_data_small = ut.collate_images_rectangular_color(all_tdata_cat.data * masks_conv_small, N=Nims, ncols=ncols, L1=320, L2=456)


recons_conv = recons_conv.data * (1 - masks_conv.float())  + all_tdata_cat[:masks_conv.size(0)].data * masks_conv.float()
recons_conv_data = ut.collate_images_rectangular_color(recons_conv, N=Nims, ncols=ncols, L1=320, L2=456)


recons_conv_small = recons_conv_small.data * (1 - masks_conv_small.float())  + all_tdata_cat[:masks_conv.size(0)].data * masks_conv_small.float()
recons_conv_data_small = ut.collate_images_rectangular_color(recons_conv_small, N=Nims, ncols=ncols, L1=320, L2=456)



recons_pca = recons_pca.data.cpu() * (1 - masks.float())  + all_tdata_cat.data * masks.float()
recons_pca_data = ut.collate_images_rectangular_color(recons_pca, N=Nims, ncols=ncols, L1=320, L2=456)

recons_pca_small = recons_pca_small.data.cpu() * (1 - masks_conv_small.float())  + all_tdata_cat.data * masks_conv_small.float()
recons_pca_data_small = ut.collate_images_rectangular_color(recons_pca_small, N=Nims, ncols=ncols, L1=320, L2=456)



all_data = torch.cat([real_data, masked_data, recons_conv_data, recons_pca_data, masked_data_small, recons_conv_data_small, recons_pca_data_small], 0)

plt.figure(figsize=(40, 20), dpi=100)
plt.imshow(all_data*0.5 + 0.5)
b=50
plt.yticks([150, 450, 750, 1050, 1350+b, 1650+b, 1950+b], ['Data', 'Missing \n Data \n R100', 'IML-CNN \n R100', 'IML-PCA\nR100','Missing \n Data\nR40', 'IML-CNN\nR40', 'IML-PCA\nR40' ], fontsize=35)
plt.xticks([])

folder = '/your_path/GANs/mouse_project/'
plt.savefig(folder + 'mdata_unregistered.eps', format='eps')


        #hhat = torch.matmul(W.t(), ((all_tdata_cat - mean_x) * masks).view(100, -1).t()).t()

        ##mean_x = Variable(mean_xorg.cuda())
        #EP = 1 
        ##for ep in range(EP):
        #    #hhat.zero_grad()
        #recons = torch.matmul((W), hhat.t()).t() + (mean_x.view(1, -1).cuda())

        ##err = (((recons - all_tdata_cat.view(100, -1))*masks.view(100, -1).float()).abs()).mean()
        #    #if 0: # ep > 500:
        #    #    dists = ((hhat.view(100, 1, 200) - means.view(1, J, 200))**2).sum(2)
        #    #    dists = dists.min(1)[0].mean()
        #    #    err = err + dists

        #    #err.backward()
        #    #opt.step()
        #    #print(ep, err.data[0])
        #    

        #recons_show = recons.view(-1, 3, 320, 456)* (1 -masks).float() + all_tdata_cat* (masks).float()
        #recons_show[recons_show < -1] = -1
        #recons_show[recons_show > 1] = 1
        #vis.images(recons_show.cpu()*0.5 + 0.5, win='recons')
        #score = (((recons - all_tdata_cat.view(100, -1))*(1 - masks.view(100, -1).float())).abs()).mean()
        #print(score)
        #all_errs.append((score, radius))
        #
        #if not os.path.exists(folder):
        #    os.mkdir(folder)
        #pickle.dump(all_errs, open(folder + '/pcav2_' + ntype + '.pk', 'wb'))
        ##torch.save([hhat, masks, radius], folder + '/pca_hhat.t')
