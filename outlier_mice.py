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
import skimage.draw as skd

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

def add_blob_noise(data, nblobs):

    data_np = data.cpu().numpy()
    N = data.size(0)
    for n in range(N):
        #print(n)
        for nblob in range(nblobs):
            sh = np.random.randint(-100, 100)
            circle = skd.circle(200 +sh, 200 +sh, 10, shape=(456, 320))
            data_np[n, circle[0], circle[1]] = -1
    return torch.from_numpy(data_np)

shared_path = '/your_path/25um_280_to_290_slices/misc_files_from_urbana/'

folder = 'outlier'
if 1:
    train_data = all_xs_cat[:1500]
    test_data = all_xs_cat[1500:]

    mean_x = train_data.mean(0, keepdim=True)
    train_data_c = (train_data - mean_x).view(train_data.size(0), -1)

    Ureal, Sreal, _ = torch.svd(train_data_c.t())

    mdl.train(mode=False)
    mdl.eval()

    W = Ureal[:, :100] 
    pca_coefs = torch.matmul(W.t(), train_data_c.t()).t()

    GMM_realdata = mix.GaussianMixture(n_components=10, covariance_type='full', tol=1e-4, 
                                      verbose=1, n_init=10)
    GMM_realdata.fit(pca_coefs.cpu().numpy())

    z_testdata = torch.matmul(W.t(), (test_data - mean_x).view(test_data.size(0),-1).t()).t()
    GMM_scores_test = GMM_realdata.score(z_testdata.cpu().numpy())

    all_outscores_pca = []
    nsample = 1

    for nblobs in range(2, 3):
        print(nblobs)
        scores = []
        for ns in range(nsample):
            outlier_data = add_blob_noise(test_data, nblobs).cuda()    

            z_outdata = torch.matmul(W.t(), (outlier_data - mean_x).view(test_data.size(0),-1).t()).t()
            scores.append(GMM_realdata.score(z_outdata.cpu().numpy()))
        all_outscores_pca.append( (scores, nblobs))

    
    pdb.set_trace()
    plt.figure(figsize=(4, 3), dpi=100)
    plt.imshow(outlier_data[0].cpu().t())
    plt.xticks([])
    plt.yticks([])
    plt.savefig('/your_path/GANs/mouse_project/outlier_example.eps')
    pdb.set_trace()


        #ims=ut.collate_images_rectangular(outlier_data[:16], N=16, ncols=4, L1=456, L2=320)
        #vis.heatmap(ims, win='out_data')

    if not os.path.exists(folder):
            os.mkdir(folder)
    #pickle.dump([all_outscores_pca, GMM_scores_test], open(folder + '/pca.pk', 'wb'))

if 0:
    train_coefs = all_hhats_cat[:1500]
    test_coefs = all_hhats_cat[1500:]
    test_data = all_xs_cat[1500:]

    mdl.train(mode=False)
    mdl.eval()

    GMM_realdata = mix.GaussianMixture(n_components=10, covariance_type='full', tol=1e-4, 
                                      verbose=1, n_init=10)
    GMM_realdata.fit(train_coefs.cpu().numpy())

    GMM_scores_test = GMM_realdata.score(test_coefs.cpu().numpy())

    all_outscores_conv = []
    nsample = 5

    for nblobs in range(1, 5):
        print(nblobs)
        scores = []
        for ns in range(nsample):
            outlier_data = add_blob_noise(test_data, nblobs).cuda()    

            z_outdata = nn.parallel.data_parallel(mdl.encoder, Variable(outlier_data.unsqueeze(1)), range(arguments.num_gpus))
            scores.append(GMM_realdata.score(z_outdata.data.squeeze().cpu().numpy()[:, :100]))
        all_outscores_conv.append( (scores, nblobs))

    if not os.path.exists(folder):
            os.mkdir(folder)
    pickle.dump([all_outscores_conv, GMM_scores_test], open(folder + '/convnet.pk', 'wb'))

