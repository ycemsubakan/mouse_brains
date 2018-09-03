import numpy as np
import torch
from algorithms_v2 import netD, netG, adversarial_trainer, VAE, NF_changedim, compute_nparam_density, conv_autoenc_mice, netg_dcgan, netd_dcgan, conv_VAE_mouse, conv_VAE_mouse_v2, adversarial_wasserstein_trainer, netd_dcgan_par, netg_dcgan_par, weights_init_seq, get_scores
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
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=24, shuffle=False,
                                           pin_memory=True, num_workers=arguments.num_gpus)


compute_kdes = 0
compute_lbp = 1

results = []
mmds = []
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
        
mdl.train(mode=False)
mdl.eval()

method = 'tsne'
if method == 'tsne':

    for i, perp in enumerate(range(95, 190, 10)):
        print(i)
        imap = mnf.TSNE(perplexity=perp, learning_rate=10, init='pca', verbose=1)

        embeds = imap.fit_transform(all_hhats_cat.squeeze().cpu().numpy())

        #all_ims_list = [im.squeeze().cpu().numpy() for im in all_ims]
        all_ims_np = all_xhats_cat.squeeze().cpu().numpy()

        plt.figure(figsize=(36, 36), dpi=100)

        #plt.subplot(1,5,i+1)
        plt.plot(embeds[:, 0], embeds[:, 1], 'o', markersize=12)
        plt.title('Perplexity {}'.format(i))

        ax = plt.gca()

        all_ims_ss = [all_ims_np[0]]
        embeds_ss = [embeds[0:1]]
        rnge = embeds[:, 0].max() - embeds[:, 0].min()

        #and (embed[0] > (embeds[:, 0].min()) + 0.1*rnge) \

        all_embeds_cat = np.concatenate(embeds_ss, 0)
        for i, (im, embed) in enumerate(zip(all_ims_np, embeds)):
            if (i % 2 == 0) \
               and np.abs(embed - all_embeds_cat).sum(1).min() > 10:
                all_ims_ss.append(im.transpose())
                embeds_ss.append(np.expand_dims(embed, 0))
                all_embeds_cat = np.concatenate(embeds_ss, 0)

        ut.plot_embedding(embeds_ss, all_ims_ss, ax, sz=(320, 456))

        plt.savefig(shared_path + 'scatter_plot_{}_perplexity_{}.png'.format(method, perp), format='png')
        plt.savefig('mouse_results/scatter_plot_{}_perplexity_{}.png'.format(method, perp), format='png')


else:
    imap = mnf.Isomap()

