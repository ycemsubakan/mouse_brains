import numpy as np
import torch
from algorithms_v2 import netD, netG, adversarial_trainer, VAE, NF_changedim, compute_nparam_density, conv_autoenc_mice, netg_dcgan, netd_dcgan, conv_VAE_mouse, conv_VAE_mouse_v3, adversarial_wasserstein_trainer, netd_dcgan_par, netg_dcgan_par, weights_init_seq, get_scores
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

#vis = visdom.Visdom(port=0, server='http://yourserver', env='your_env2')
#assert vis.check_connection()

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='choose your model ', default='NF')
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
elif hostname == 'cem-gpu':
    train_data_dir = '/home/cem/unregistered_images/'


transform = transforms.Compose([
        #transforms.Grayscale(),
        transforms.Resize(size=(320, 456)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
dset_train = datasets.ImageFolder(train_data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=24, shuffle=True,
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
    Ks = [200, 200]
    L = 456*320
        
    mdl = conv_VAE_mouse_v3(320, 456, Ks, M=64, num_gpus=arguments.num_gpus) 
       
    if arguments.cuda:
        mdl.cuda()

    path = 'models/VAE_arc5_l1_{}_K_{}.t'.format(arguments.data, Ks)
    if 1 & os.path.exists(path):
        mdl.load_state_dict(torch.load(path))

nbatches = 75 
hhat_path = 'mouse_embeddings/' + 'hats_unregistered.t'
if 1 & os.path.exists(hhat_path):
    all_xs = []
    dcts = torch.load(hhat_path)

    #all_xs_cat = torch.cat(all_xs, dim=0)
    all_hhats_cat = dcts['hhats']
    all_xhats_cat = dcts['xhats']

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
    #all_xhats_cat = torch.cat(all_xhats, dim=0) 
    #all_xs_cat = torch.cat(all_xs, dim=0)

    dct = {'hhats' : all_hhats_cat, 
           'xhats' : []}
    if not os.path.exists('mouse_embeddings'):
        os.mkdir('mouse_embeddings')
    torch.save(dct, hhat_path)

shared_path = '/your_path/25um_280_to_290_slices/misc_files_from_urbana/'

mean_hhat = all_hhats_cat.mean(0, keepdim=True)
all_hhats_cat_c = all_hhats_cat - mean_hhat

U, S, V = torch.svd(all_hhats_cat_c.t())

#mean_x = all_xs_cat.mean(0, keepdim=True)
#all_xs_cat_c = (all_xs_cat - mean_x).view(all_xs_cat.size(0), -1)

#Ureal, _, _ = torch.svd(all_xs_cat_c.t())
plt.figure(figsize=(42, 20), dpi=120)

tweak_mode = 2

mdl.train(mode=False)
mdl.eval()

alpha = torch.arange(-10, 10, 2)

tweaks = []
for k in range(10):
    acts = (U[:, k].unsqueeze(1) * alpha.unsqueeze(0).cuda()).t() + mean_hhat
    acts = acts.unsqueeze(-1).unsqueeze(-1)
    tweaks.append(mdl.decoder(Variable(acts)).data)

tweaks_cat = torch.cat(tweaks, dim=0)
tweaks_cat = torch.cat([tweaks_cat, -torch.ones(tweaks_cat.size(0), 1, 320, 456).cuda()], dim=1)

im1 = ut.collate_images_rectangular_color(tweaks_cat.data, 100, 10, L1=320, L2=456)

plt.subplot2grid((1,11),(0,0), colspan=9)
plt.imshow(0.5*im1 + 0.5)

plt.xticks([])
plt.yticks([])


#plt.figure(figsize=(30, 20), dpi=120) 
#torchvision.utils.save_image(0.5*tweaks_cat.data.cpu() + 0.5, '/your_path/GANs/mouse_project/tweak_images.png', nrow=10, padding=2)
#plt.imshow(0.5*im + 0.5, interpolation='None')
    #plt.savefig('mouse_results/tweak_images_unregistered.png', format='png')
    
    
#nrrd.write(shared_path + 'tweak_images.nrrd', tweaks.data.cpu().squeeze().permute(1, 2, 0).numpy())

alpha = torch.randn(80)*10

Ncomps = 10 
#plt.figure(figsize=(40, 10), dpi=120)

all_acts = []
all_tweaks = []
all_real_comps = []
for k in range(Ncomps):
    print(k)
    acts = (U[:, k].unsqueeze(1) * alpha.unsqueeze(0).cuda()).t() + mean_hhat
    outs = nn.parallel.data_parallel(mdl.decoder, Variable(acts.unsqueeze(-1).unsqueeze(-1)), range(arguments.num_gpus))
    all_tweaks.append(outs.mean(1).std(0).data)

all_st_tweaks_cat = torch.cat(all_tweaks, dim=0)

plt.subplot2grid((1,11), (0, 9))
plt.imshow(all_st_tweaks_cat)
plt.xticks([])
plt.yticks([])
    #plt.subplot(2, 5, k+1)
    #plt.imshow(all_tweaks[k].cpu().squeeze().t().numpy(), interpolation='None')
#plt.title('Component {}'.format(k+1), fontsize=25)
#plt.xticks([])
#plt.yticks([])

#all_tweaks_cat = torch.cat(all_tweaks, dim=0).squeeze().permute(1,2,0).cpu().numpy()
#nrrd.write(shared_path + 'tweak_images.nrrd', all_tweaks_cat)

plt.tight_layout()
plt.savefig('/your_path/GANs/mouse_project/tweak_images_unregistered_v2.png', format='png')
#plt.savefig(shared_path + 'tweak_images_v2.eps', format='eps')

pdb.set_trace()

#    plt.figure(figsize=(24, 16), dpi=120)
#    for k in range(Ncomps):
#        st_image = (Ureal[:, k].view(1, 456, 320)*(alpha.view(-1, 1, 1).cuda()) + mean_x).std(0) 
#        all_real_comps.append(st_image)
#
#        plt.subplot(4, 5, k+1)
#        plt.imshow(all_real_comps[k].cpu().t().numpy(), interpolation='None')
#        plt.title('Component {}'.format(k+1))
#        plt.xticks([])
#        plt.yticks([])
#
#    plt.savefig('mouse_results/pca_realdata_correct.png', format='png')
#    plt.savefig(shared_path + 'pca_realdata_correct.eps', format='eps')
#
#


#vis.heatmap(im, win='tweaks', opts=opts)
GMM = mix.GaussianMixture(n_components=30, covariance_type='full', tol=1e-4, 
                                  verbose=1, n_init=5)
GMM.fit(all_acts_cat.squeeze().cpu().numpy()) 

seed = torch.from_numpy(GMM.sample(200)[0]).float().cuda()
gen_data = nn.parallel.data_parallel(mdl.decoder, Variable(seed.unsqueeze(-1).unsqueeze(-1)) , range(arguments.num_gpus))

plt.figure(figsize=(15, 3), dpi=200)

plt.subplot(1,3,1)
std_realimages = all_xs_cat.std(0).sqrt()
plt.imshow(std_realimages.cpu().squeeze().t().numpy(), interpolation='None')
plt.title('real images st. deviation')
plt.clim(vmin=0, vmax=std_realimages.max())
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(1,3,2)
std_reconstructions = all_xhats_cat.std(0).sqrt()
plt.imshow(std_reconstructions.cpu().squeeze().t().numpy(), interpolation='None')
plt.title('reconstruction images st. deviation')
plt.clim(vmin=0, vmax=std_realimages.max())
plt.xticks([])
plt.yticks([])

plt.colorbar()

plt.subplot(1,3,3)
std_genimages = gen_data.std(0).sqrt()
plt.imshow(std_genimages.data.cpu().squeeze().t().numpy(), interpolation='None')
plt.title('generated images st. deviation')
plt.clim(vmin=0, vmax=std_realimages.max())
plt.xticks([])
plt.yticks([])

plt.colorbar()

plt.savefig('mouse_results/stdeviation_maps.png', format='png')
if not os.path.exists(shared_path):
    os.mkdir(shared_path)
plt.savefig(shared_path + 'stdeviation_maps.eps', format='eps')


im = ut.collate_images_rectangular(gen_data.data, 64, 8, L1=456, L2=320)
#opts = {'title':'gen_data'}
#vis.heatmap(im, win='xgen', opts=opts)

plt.figure(figsize=(20, 28), dpi=120)
plt.imshow(im, interpolation='None')
plt.savefig('mouse_results/generated_images.png', format='png')

opts = {'title':'all_hhat'}
vis.heatmap(all_hhats_cat[:100].cpu(), win='hhat', opts=opts)

writepath = '/your_path/25um_280_to_290_slices' + '/currentstateofimages/' + model_desc
if not os.path.exists( writepath ):
    os.mkdir(writepath)

f = open(writepath + '/model_specification.txt', 'w')
f.write('encoder:' + str(mdl.__dict__['_modules']['encoder']) + '\n')
f.write('decoder:' + str(mdl.__dict__['_modules']['decoder']) + '\n')
f.close()

N = 200

time_stamp = str(round(time.time()))
all_ims = np.zeros((456, 320, N*2))

print('getting xs')
all_ims[:, :, :N] = all_xs_cat[:N].permute(1, 2, 0).cpu().numpy().squeeze()

print('getting gen_data')
all_ims[:, :, N:2*N] = gen_data[:N].data.squeeze().cpu().permute(1, 2, 0).numpy()

#nrrd.write(writepath + '/real_plus_generated_' + time_stamp + '.nrrd', all_ims)

print('getting xhats')
all_ims[:, :, N:2*N] = all_xhats_cat[:N].cpu().permute(1, 2, 0).numpy().squeeze()
#nrrd.write(writepath + '/real_plus_reconstructed_' + time_stamp + '.nrrd', all_ims)

# save xhats and generated data
print('saving generated and reconstruction data')
torch.save(gen_data, open('/your_root/data2/mice_gen_randoms_v2.t', 'wb'))
torch.save(all_xhats_cat, open('/your_root/data2/mice_reconstructions.t', 'wb'))

