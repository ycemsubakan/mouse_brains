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
import torchvision

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

dset_test = datasets.ImageFolder(train_data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=24, shuffle=False,
                                          pin_memory=True, num_workers=arguments.num_gpus)

for i, (unregdata, _) in enumerate(it.islice(train_loader, 0, 1, 1)):
    pass
unregdata[:, 2, :, :] = -1

torchvision.utils.save_image(unregdata[:16], '/your_path/GANs/mouse_project/poster/unregistered_16.png', nrow=4)

pdb.set_trace()



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

folder = '/your_path/GANs/mouse_project/'

##### pca 
mean_x = all_xs_cat.mean(0, keepdim=True)
all_xs_cat_c = (all_xs_cat - mean_x).view(all_xs_cat.size(0), -1)

Ureal, Sreal, _ = torch.svd(all_xs_cat_c.t().cpu())

W = Ureal[:, :500].cuda()
pca_coefs = torch.matmul(W.t(), all_xs_cat_c.t()).t()

J = 56
GMM = mix.GaussianMixture(n_components=J, covariance_type='full', tol=1e-4, 
                                  verbose=1, n_init=10)
GMM.fit(pca_coefs.squeeze().cpu().numpy()) 

seed = torch.from_numpy(GMM.sample(16)[0]).float().cuda()

gen_data = torch.matmul(W, seed.t()).t().contiguous().view(-1, 2, 320, 456) + mean_x
gen_data[gen_data>1] = 1
gen_data[gen_data<-1] = -1
gen_data = torch.cat([gen_data, -torch.ones(gen_data.size(0), 1, 320, 456).cuda()], dim=1)

Nims = 10
ncols = 2
collated_gen_data_pca = ut.collate_images_rectangular_color(gen_data, Nims, ncols=ncols, L1=320, L2=456)

##conv net
mdl.train(mode=False)
mdl.eval()

GMM = mix.GaussianMixture(n_components=J, covariance_type='full', tol=1e-4, 
                                  verbose=1, n_init=10)
GMM.fit(all_hhats_cat.squeeze().cpu().numpy()) 
seed = torch.from_numpy(GMM.sample(16)[0]).float().cuda()

gen_data = nn.parallel.data_parallel(mdl.decoder, Variable(seed.unsqueeze(-1).unsqueeze(-1)) , range(arguments.num_gpus)).data
gen_data = torch.cat([gen_data, -torch.ones(gen_data.size(0), 1, 320, 456).cuda()], dim=1)
collated_gen_data_conv = ut.collate_images_rectangular_color(gen_data, Nims, ncols=ncols, L1=320, L2=456)


## conv net VAE
gen_data, seed = mdl.generate_data(16)
gen_data = gen_data.data
gen_data = torch.cat([gen_data, -torch.ones(gen_data.size(0), 1, 320, 456).cuda()], dim=1)
collated_gen_data_VAE = ut.collate_images_rectangular_color(gen_data, Nims, ncols=ncols, L1=320, L2=456)


# add the third dim and get the image
all_xs_cat = torch.cat([all_xs_cat, -torch.ones(all_xs_cat.size(0), 1, 320, 456).cuda()], dim=1)
collated_real_data = ut.collate_images_rectangular_color(all_xs_cat, Nims, ncols=ncols, L1=320, L2=456)

seper = torch.ones(collated_real_data.size(0), 10, 3)
all_ims_at_one = torch.cat([collated_real_data, seper, collated_gen_data_VAE, seper, collated_gen_data_conv, seper, collated_gen_data_pca], dim=1).cpu().numpy()

pdb.set_trace()

plt.figure(figsize=(24, 20), dpi=100)
plt.imshow(all_ims_at_one*0.5 + 0.5)
plt.xticks([350, 1400, 2200, 3000], ['Real Data', 'VAE-CNN', 'IML-CNN', 'IML-PCA'], fontsize=35)
plt.yticks([])

plt.savefig(folder + 'gen_unregistered_v2.eps', format='eps')

