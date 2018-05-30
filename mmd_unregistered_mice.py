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

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev2')
assert vis.check_connection()

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='choose your model bitch', default='NF')
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
    train_data_dir = '/media/data2/unregistered_images/'
    test_data_dir = '/media/data2/unregistered_images_test/'
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

dset_test = datasets.ImageFolder(train_data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=24, shuffle=False,
                                          pin_memory=True, num_workers=arguments.num_gpus)


# load the model
Ks = [200, 200]
    
mdl = conv_VAE_mouse_v3(320, 456, Ks, M=64, num_gpus=arguments.num_gpus) 
   
if arguments.cuda:
    mdl.cuda()

model_desc = 'VAE_arc3_l1_lrelu_d48'
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


folder = 'unregistered_results'
##### do pca see what happens

if arguments.dopca:
    mean_x = all_xs_cat.mean(0, keepdim=True)
    all_xs_cat_c = (all_xs_cat - mean_x).view(all_xs_cat.size(0), -1)

    Ureal, Sreal, _ = torch.svd(all_xs_cat_c.t().cpu())

    W = Ureal[:, :200].cuda()
    pca_coefs = torch.matmul(W.t(), all_xs_cat_c.t()).t()

    recons = torch.matmul(W.cpu(), pca_coefs[:100].cpu().t()).t().contiguous().view(-1, 3, 320, 456) + mean_x.cpu()

    recons[recons>1] = 1
    recons[recons<-1] = -1

    opts = {'title' : 'pca reconstructions'} 
    vis.images(recons*0.5 + 0.5, opts=opts) 

    all_mmds = []
    all_stats = []
    num_samples = 5
    for J in range(1, 60, 5):
        print(J)
        GMM = mix.GaussianMixture(n_components=J, covariance_type='full', tol=1e-4, 
                                          verbose=1, n_init=10)
        GMM.fit(pca_coefs.squeeze().cpu().numpy()) 

        mmds = [] 
        for n in range(num_samples):
            print(n)
            seed = torch.from_numpy(GMM.sample(500)[0]).float().cuda()

            gen_data = torch.matmul(W, seed.t()).t().contiguous().view(-1, 3, 320, 456) + mean_x
            gen_data[gen_data>1] = 1
            gen_data[gen_data<-1] = -1

            mmds.append(compute_mmd(gen_data.view(gen_data.size(0), -1), all_xs_cat.view(all_xs_cat.size(0), -1), cuda=arguments.cuda, kernel='linear', sig=1))

        all_mmds.append( (mmds, J) )
        all_stats.append( (np.mean(mmds), np.std(mmds), J) )
        print(all_mmds)
        print(all_stats)

    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump([all_stats, all_mmds], open(folder + '/pca.pk', 'wb'))
    
    #opts = {'title' : 'pca randoms'}
    #vis.images(gen_data.cpu()*0.5 + 0.5, opts=opts) 
    pdb.set_trace()


mdl.train(mode=False)
mdl.eval()

#vis.images(gen_data.data.cpu()*0.5 + 0.5) 
if 0:
    print('evaluating conv net..')
    all_mmds = []
    all_stats = []
    num_samples = 5
    for J in range(1, 60, 5):
        print(J)
        GMM = mix.GaussianMixture(n_components=J, covariance_type='full', tol=1e-4, 
                                          verbose=1, n_init=10)
        GMM.fit(all_hhats_cat.squeeze().cpu().numpy()) 

        mmds = [] 
        for n in range(num_samples):
            print(n)
            seed = torch.from_numpy(GMM.sample(500)[0]).float().cuda()

            gen_data = nn.parallel.data_parallel(mdl.decoder, Variable(seed.unsqueeze(-1).unsqueeze(-1)) , range(arguments.num_gpus)).data

            mmds.append(compute_mmd(gen_data.view(gen_data.size(0), -1), all_xs_cat.view(all_xs_cat.size(0), -1), cuda=arguments.cuda, kernel='linear', sig=1))

        all_mmds.append( (mmds, J) )
        all_stats.append( (np.mean(mmds), np.std(mmds), J) )
        print(all_mmds)
        print(all_stats)

    if not os.path.exists(folder):
            os.mkdir(folder)
    pickle.dump([all_stats, all_mmds], open(folder + '/conv_net.pk', 'wb'))


if 1:
    print('computing vae scores..')
    num_samples=5
    mdl.train(mode=False)
    mdl.eval()

    mmds = [] 
    for n in range(num_samples):
        print(n)
        seed = torch.randn(500, 200).cuda()

        gen_data = nn.parallel.data_parallel(mdl.decoder, Variable(seed.unsqueeze(-1).unsqueeze(-1)) , range(arguments.num_gpus)).data

        mmds.append(compute_mmd(gen_data.view(gen_data.size(0), -1), all_xs_cat.view(all_xs_cat.size(0), -1), cuda=arguments.cuda, kernel='linear', sig=1))

    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(mmds, open(folder + '/vae.pk', 'wb'))

