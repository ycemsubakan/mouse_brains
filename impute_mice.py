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
if 0:
    mean_x = all_xs_cat.mean(0, keepdim=True)

    all_xs_cat_c = (all_xs_cat - mean_x).view(all_xs_cat.size(0), -1)

    Ureal, Sreal, _ = torch.svd(all_xs_cat_c.t().cpu())

    W = Ureal[:, :200].cuda()
    pca_coefs = torch.matmul(W.t(), all_xs_cat_c.t()).t()

    J = 30
    GMM = mix.GaussianMixture(n_components=J, covariance_type='full', tol=1e-4, 
                                          verbose=1, n_init=10)
    GMM.fit(pca_coefs.squeeze().cpu().numpy()) 
    means = torch.from_numpy(GMM.means_).float()
    means = Variable(means.cuda())
    
    for radius in range(40, 41, 20):
    # get the masks right
        if ntype == 'spec':
            p = 0.5

            masks = torch.rand(100, 2, 320, 456) < p
            
        else:
            sh = np.random.randint(-20, 20)
            masks = torch.ones(100, 2, 320, 456)
            for n in range(100):
                print(n)
                circle = skd.circle(100 +sh, 200 +sh, radius, shape=(320, 456))
                im = torch.ones(2, 320, 456)
                im[:, circle[0], circle[1]] = 0
                masks[n] = im #torch.from_numpy(im)

        masks = Variable(masks.cuda())
        
        all_tdata = []
        for tdata, _ in test_loader:
            if 1:
                tdata = tdata[:, :2, :, :]

            all_tdata.append(tdata)
        all_tdata_cat = Variable(torch.cat(all_tdata, 0).cuda())

        Ntest = 80
        masks = masks[:Ntest]
        all_tdata_cat = all_tdata_cat[:Ntest]

        hhat = Variable(torch.zeros(Ntest, 200).cuda(), requires_grad=True)

        opt = torch.optim.Adam([hhat], betas=(0.5, 0.999), lr=0.05)
        #opt = torch.optim.SGD([hhat], lr=0.001)

        #mean_x = Variable(mean_xorg.cuda())
        EP = 2000
        for ep in range(EP):
            #hhat.zero_grad()
            recons = torch.matmul(Variable(W), hhat.t()).t() + Variable(mean_x.view(1, -1).cuda())

            err = (((recons - all_tdata_cat.view(Ntest, -1))*masks.view(Ntest, -1).float()).abs()).mean()
            if 0: # ep > 500:
                dists = ((hhat.view(100, 1, 200) - means.view(1, J, 200))**2).sum(2)
                dists = dists.min(1)[0].mean()
                err = err + dists

            err.backward()
            opt.step()
            print(ep, err.item())
            

            if 0 and (ep % 500 == 0):
                recons_show = recons.view(-1, 3, 320, 456).data * (1 -masks.data).float() + all_tdata_cat.data * (masks.data).float()
                recons_show[recons_show < -1] = -1
                recons_show[recons_show > 1] = 1
                vis.images(recons_show.cpu()*0.5 + 0.5, win='recons')
        score = (((recons - all_tdata_cat.view(Ntest, -1))*(1 - masks.view(Ntest, -1).float())).abs()).mean()
        print(score.item())
        all_errs.append((score.data.cpu().item(), radius))
        
        if not os.path.exists(folder):
            os.mkdir(folder)
        #pickle.dump(all_errs, open(folder + '/pcav2_' + ntype + '.pk', 'wb'))
        if radius == 40:
            torch.save([hhat.data, masks.data, radius], folder + '/pca_hhat_rad{}.t'.format(radius))

if 1:
        
    for radius in range(40, 41, 20):
        # get the masks right
        if ntype == 'spec':
            p = 0.5

            masks = torch.rand(100, 2, 320, 456) < p
            
        else:
            sh = np.random.randint(-20, 20)
            masks = torch.ones(100, 2, 320, 456)
            for n in range(100):
                print(n)
                circle = skd.circle(100 +sh, 200 +sh, radius, shape=(320, 456))
                im = torch.ones(2, 320, 456)
                im[:, circle[0], circle[1]] = 0
                masks[n] = im #torch.from_numpy(im)

        masks = Variable(masks.cuda())

        all_tdata = []
        for tdata, _ in test_loader:
            tdata = tdata[:, :2, :, :]
            all_tdata.append(tdata)
        all_tdata_cat = Variable(torch.cat(all_tdata, 0).cuda())

        Ntest = 80
        masks = masks[:Ntest]
        all_tdata_cat = all_tdata_cat[:Ntest]

        hhat = Variable(torch.zeros(Ntest, 200).cuda(), requires_grad=True)

        opt = torch.optim.Adam([hhat], betas=(0.5, 0.999), lr=0.001)
        #opt = torch.optim.SGD([hhat], lr=0.001)

        #mean_x = Variable(mean_xorg.cuda())
        EP = 100
        for ep in range(EP):
            #hhat.zero_grad()
            #recons = torch.matmul(Variable(W), hhat.t()).t() + Variable(mean_x.view(1, -1).cuda())
            recons = nn.parallel.data_parallel(mdl.decoder, (hhat.unsqueeze(-1).unsqueeze(-1)) , range(arguments.num_gpus))

            err = (((recons - all_tdata_cat)*masks.float()).abs()).mean()
            if 0: # ep > 500:
                dists = ((hhat.view(100, 1, 200) - means.view(1, J, 200))**2).sum(2)
                dists = dists.min(1)[0].mean()
                err = err + dists

            err.backward()
            opt.step()
            print(ep, err.cpu().item())
            

            if 0 and (ep % 25) == 0:
                recons_show = recons.view(-1, 3, 320, 456).data * (1 -masks.data).float() + all_tdata_cat.data * (masks.data).float()
                recons_show[recons_show < -1] = -1
                recons_show[recons_show > 1] = 1
                vis.images(recons_show.cpu()*0.5 + 0.5, win='recons')
        score = (((recons - all_tdata_cat)*(1 - masks.float())).abs()).mean()
        all_errs.append((score.data.cpu().item(), radius))
        
        if not os.path.exists(folder):
            os.mkdir(folder)

        if radius == 40:
            torch.save([hhat.data, masks.data, radius], folder + '/conv_net_hhat_rad{}.t'.format(radius))
        #pickle.dump(all_errs, open(folder + '/convnet_' + ntype + '.pk', 'wb'))

    
    
    
