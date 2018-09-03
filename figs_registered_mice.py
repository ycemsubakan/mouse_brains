import numpy as np
import torch
from algorithms_v2 import netD, netG, adversarial_trainer, VAE, NF_changedim, compute_nparam_density, conv_autoenc_mice, netg_dcgan, netd_dcgan, conv_VAE_mouse, conv_VAE_mouse_v2, adversarial_wasserstein_trainer, netd_dcgan_par, netg_dcgan_par, weights_init_seq, get_scores, compute_mmd
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

vis = visdom.Visdom(port=0, server='http://yourserver', env='your_env2')
assert vis.check_connection()

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='choose your model ', default='NF')
arguments = argparser.parse_args()

np.random.seed(2)
torch.manual_seed(9)
arguments.cuda = torch.cuda.is_available()
arguments.batch_size = 128
arguments.data = 'mice_norm2'
arguments.input_type = 'autoenc'

home = os.path.expanduser('~')
hostname = socket.gethostname()

if hostname == 'nmf':
    train_data_dir = '/your_root/data2/mice_data.t'
    mask_dir = '/your_root/data2/mice_data_masks.t'
elif hostname == 'cem-gpu':
    train_data_dir = home + '/mouse_data/mice_data.t'
    mask_dir = home + '/mouse_data/mice_data_masks.t'


data = torch.load(open(train_data_dir, 'rb'))
data = data.unsqueeze(1)

masks = torch.load(open(mask_dir, 'rb'))[0]

if arguments.data == 'mice':
    im_max = 400
    data[data > im_max] = im_max
    data = data / im_max
    data = 2*data - 1
else:
    im_max = 800
    data[data > im_max] = im_max

    data = data**0.5

    vecims = data.view(data.size(0), -1)
    im_max = vecims.max(1)[0].view(-1, 1, 1, 1)

    data = data / im_max
    data = 2*data - 1

dset_train = torch.utils.data.TensorDataset(data, data)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=24, shuffle=True,
                                           pin_memory=True, num_workers=arguments.num_gpus)

compute_kdes = 0
compute_lbp = 1

results = []
model = arguments.model
if model == 'NF':
    EP = 25
    base_inits = 10

    N, M = 28, 28

    # now fit
    base_dist = 'mixture_full_gauss'
    Kdict = 200

    Kss = [[100]]
    mdl = conv_autoenc_mice(base_inits=base_inits, K=Ks[0], Kdict=Kdict,
                            num_gpus=arguments.num_gpus)
    if arguments.cuda:
        mdl.cuda()

    #path = 'models/convauto_nobatch_reluft_{}_K_{}.t'.format(arguments.data, Ks)
    path = 'models/convauto_{}_K_{}.t'.format(arguments.data, Ks)

    if 1 and os.path.exists(path):
        mdl.load_state_dict(torch.load(path))
        
            
elif model == 'VAE': 
    EP = 25
    Ks = [100, 100]
    L = 456*320
        
    mdl = conv_VAE_mouse_v2(L, L, Ks, M=64, num_gpus=arguments.num_gpus) 
       
    if arguments.cuda:
        mdl.cuda()

    model_desc = 'VAE_arc3_l1'
    path = 'models/' + model_desc + '_{}_K_{}.t'.format(arguments.data, Ks)
    if 1 & os.path.exists(path):
        mdl.load_state_dict(torch.load(path))

nbatches = 75 
hhat_path = 'mouse_embeddings/' + 'hats.t'
if 1 & os.path.exists(hhat_path):
    all_xs = []
    for i, (data, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
        if arguments.cuda:
            data = data.cuda()

        all_xs.append(data.squeeze())
        print(i)
    dcts = torch.load(hhat_path)

    all_xs_cat = torch.cat(all_xs, dim=0)
    all_hhats_cat = dcts['hhats']
    all_xhats_cat = dcts['xhats']

else:
    all_hhats = []
    all_xhats = []
    all_xs = []
    for i, (data, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
        if arguments.cuda:
            data = data.cuda()

        hhat = nn.parallel.data_parallel(mdl.encoder, Variable(data), range(arguments.num_gpus))
        xhat = nn.parallel.data_parallel(mdl.decoder, hhat[:, :100], range(arguments.num_gpus))
        all_hhats.append(hhat.data.squeeze())
        all_xhats.append(xhat.data.squeeze())
        all_xs.append(data.squeeze())
        print(i)
    all_hhats_cat = torch.cat(all_hhats, dim=0)[:, :100]
    all_xhats_cat = torch.cat(all_xhats, dim=0) 
    all_xs_cat = torch.cat(all_xs, dim=0)

    dct = {'hhats' : all_hhats_cat, 
           'xhats' : all_xhats_cat}
    if not os.path.exists('mouse_embeddings'):
        os.mkdir('mouse_embeddings')
    torch.save(dct, hhat_path)

shared_path = '/your_path/25um_280_to_290_slices/misc_files_from_urbana/'

folder = 'registered_results'
if 0:
    mean_x = all_xs_cat.mean(0, keepdim=True)
    all_xs_cat_c = (all_xs_cat - mean_x).view(all_xs_cat.size(0), -1)

    Ureal, Sreal, _ = torch.svd(all_xs_cat_c.t())

    mdl.train(mode=False)
    mdl.eval()

    W = Ureal[:, :500] 
    pca_coefs = torch.matmul(W.t(), all_xs_cat_c.t()).t()

    recons = torch.matmul(W, pca_coefs.t()).t().contiguous().view(-1, 456, 320) + mean_x

    J = 30
    GMM_realdata = mix.GaussianMixture(n_components=J, covariance_type='full', tol=1e-4, 
                                          verbose=1, n_init=10)
    GMM_realdata.fit(pca_coefs.cpu().numpy())

    random_coefs = torch.from_numpy(GMM_realdata.sample(n_samples=30)[0]).float().cuda()
    random_pcadata = torch.matmul(W, random_coefs.t()).t().contiguous().view(-1, 456, 320) + mean_x

    random_pcadata[random_pcadata > 1] = 1
    random_pcadata[random_pcadata < -1] = -1

    ims = ut.collate_images_rectangular(random_pcadata.permute(0, 2, 1), 16, ncols=4, L1=320, L2=456)
    ims_real = ut.collate_images_rectangular(all_xs_cat.permute(0, 2, 1), 16, ncols=4, L1=320, L2=456)

    #vis.heatmap(ims, win='rpca')
    ims = ims.numpy()
    imreal_np = ims_real.numpy()
    
    #random_pcadata_np = random_pcadata.squeeze().permute(1, 2, 0).cpu().numpy()
    pdb.set_trace()

    plt.figure(figsize=(12, 16), dpi=120)
    plt.imshow(np.flipud(ims))
    plt.savefig('/your_path/GANs/mouse_project/highres_pca.eps', format='eps')

    plt.figure(figsize=(12, 16), dpi=120)
    plt.imshow(np.flipud(imreal_np))
    plt.savefig('/your_path/GANs/mouse_project/real_data_registered.eps', format='eps')

    #nrrd.write(shared_path + 'high_resolution_pca_registed.nrrd', random_pcadata_np)

    pdb.set_trace()
if 0:    

    print('computing vae scores')
    num_samples = 1
    mdl.train(mode=False)
    mdl.eval()

    mmds = [] 
    for n in range(num_samples):
        print(n)
        seed = torch.randn(500, 100).cuda()

        gen_data = nn.parallel.data_parallel(mdl.decoder, Variable(seed.unsqueeze(-1).unsqueeze(-1)) , range(arguments.num_gpus)).data

        mmds.append(compute_mmd(gen_data.view(gen_data.size(0), -1), all_xs_cat.view(all_xs_cat.size(0), -1), cuda=arguments.cuda, kernel='linear', sig=1))

    ims = ut.collate_images_rectangular(gen_data.squeeze().permute(0, 2, 1), 16, ncols=4, L1=320, L2=456).cpu().numpy()

    plt.figure(figsize=(12, 16), dpi=120)
    plt.imshow(np.flipud(ims))
    plt.savefig('/your_path/GANs/mouse_project/vae_gens_registered.eps', format='eps')



    if not os.path.exists(folder):
            os.mkdir(folder)
    pickle.dump(mmds, open(folder + '/vae.pk', 'wb'))

    #gen_data_vae = gen_data[:20].squeeze().permute(1, 2, 0).cpu().numpy()
    #nrrd.write(shared_path + 'vae_registered.nrrd', gen_data_vae)

    pdb.set_trace()




if 1:
    mdl.train(mode=False)
    mdl.eval()

    print('evaluating conv net..')
    all_mmds = []
    all_stats = []
    num_samples = 1
    for J in range(30, 31, 5):
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

    ims = ut.collate_images_rectangular(gen_data.squeeze().permute(0, 2, 1), 16, ncols=4, L1=320, L2=456).cpu().numpy()

    plt.figure(figsize=(12, 16), dpi=120)
    plt.imshow(np.flipud(ims))
    plt.savefig('/your_path/GANs/mouse_project/iml_gens_registered.eps', format='eps')



        #st_pcadata = random_pcadata.std(0).view(456, 320)
        #vis.heatmap(st_pcadata.cpu(), win='st_pca')

        #opts = {'xmin': -1, 'xmax': 1}

        #im_randompcas = ut.collate_images_rectangular(random_pcadata, 32, 4, L1=456, L2=320)
        #vis.heatmap(im_randompcas, win='random_pca', opts=opts)

 

#pca_recons = ut.collate_images_rectangular(recons, 16, 4, L1=456, L2=320)
#vis.heatmap(pca_recons, win='pca_recons')
#
#nnet_recons = ut.collate_images_rectangular(all_xhats_cat, 16, 4, L1=456, L2=320)
#vis.heatmap(nnet_recons, win='nnet_recons')
#
#real_data = ut.collate_images_rectangular(all_xs_cat, 16, 4, L1=456, L2=320)
#vis.heatmap(real_data, win='real_data')
#



