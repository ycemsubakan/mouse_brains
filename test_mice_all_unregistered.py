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
import pickle
from torchvision import datasets, transforms
import argparse
from WassersteinGAN.models.dcgan import DCGAN_G
import nrrd
import skimage.feature as skf
import socket

home = os.path.expanduser('~')
hostname = socket.gethostname()

if hostname == 'nmf':
    env = 'your_env'
elif hostname == 'cem-gpu':
    env = 'your_env2'

vis = visdom.Visdom(port=0, server='http://yourserver', env=env)
assert vis.check_connection()

# now get (generate) some data and fit 
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='choose your model', default='NF')
arguments = argparser.parse_args()

np.random.seed(2)
torch.manual_seed(9)
arguments.cuda = torch.cuda.is_available()
arguments.batch_size = 128
arguments.data = 'mice_nonregistered'
arguments.input_type = 'autoenc'


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


for dt, _ in it.islice(train_loader, 0, 1, 1):
    opts = {}
    #images = ut.collate_images_rectangular(dt, 16, 4, L1=456, L2=320)
    #vis.heatmap(images, win='samples')
    vis.images(0.5 + 0.5*dt, win='samples', nrow=4)
#h = torch.randn(100, 100, 1, 1)
#out = netG.forward(Variable(h))

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
    for config_num, Ks in enumerate(Kss):
        mdl = conv_autoenc_mice(base_inits=base_inits, K=Ks[0], Kdict=Kdict,
                           num_gpus=arguments.num_gpus)
        if arguments.cuda:
            mdl.cuda()

        #path = 'models/convauto_nobatch_reluft_{}_K_{}.t'.format(arguments.data, Ks)
        path = 'models/convauto_{}_K_{}.t'.format(arguments.data, Ks)

        if 1 and os.path.exists(path):
            
            mdl.load_state_dict(torch.load(path))
            if 1 and os.path.exists(path + '.gmm'):
                mdl.GMM = pickle.load(open(path + '.gmm', 'rb'))
            else:
                mdl.gmm_trainer(train_loader, arguments.cuda, vis=vis)  
                pickle.dump(mdl.GMM, open(path + '.gmm', 'wb'))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            mdl.trainer(train_loader, vis, EP, arguments.cuda, config_num) 
            torch.save(mdl.state_dict(), path)
            mdl.gmm_trainer(train_loader, arguments.cuda, vis=vis)  
            pickle.dump(mdl.GMM, open(path + '.gmm', 'wb'))
        gen_data, seed = mdl.generate_data(100)
        opts = {'title':'NF generated data config {}'.format(config_num)}
        #vis.images(0.5 + 0.5*gen_data.data.cpu(), opts=opts, win='NF_config_{}'.format(config_num))
        gen_ims = ut.collate_images_rectangular(gen_data.data, 20, ncols=4, L1=456, L2=320)
        vis.heatmap(gen_ims, opts=opts, win='NF_config_{}'.format(config_num))

        if compute_kdes:
            num_samples = 10
            av_lls, im_gen, im_test = compute_nparam_density(test_loader, mdl, 0.2,
                               arguments.cuda, num_samples=num_samples)
            av_mmds = get_mmds(test_loader, mdl, 10, arguments.cuda, 
                               num_samples=num_samples)

            results.append((av_lls, Ks))
            mmds.append((av_mmds, Ks))

            vis.image(im_gen*0.5 + 0.5, win='NF genim') 
            vis.image(im_test*0.5 + 0.5, win='NF testim') 


elif model == 'VAE': 
    EP = 50
    Kss = [[500, 500]]
    L = 456*320
    for config_num, Ks in enumerate(Kss):
        
        mdl = conv_VAE_mouse_v3(320, 456, Ks, M=64, num_gpus=arguments.num_gpus) 
       
        if arguments.cuda:
            mdl.cuda()

        path = 'models/VAE_arc5_l1_{}_K_{}.t'.format(arguments.data, Ks)
        if 1 & os.path.exists(path):
            mdl.load_state_dict(torch.load(path))
        else:
            if os.path.exists(path):
                mdl.load_state_dict(torch.load(path))

            if not os.path.exists('models'):
                os.mkdir('models')
            mdl.VAE_trainer(arguments.cuda, train_loader, vis=vis, 
                            EP=EP, config_num=config_num)
            torch.save(mdl.state_dict(), path)

            f = open(path + '.txt', 'w')
            f.write('encoder:' + str(mdl.__dict__['_modules']['encoder']) + '\n')
            f.write('decoder:' + str(mdl.__dict__['_modules']['decoder']) + '\n')
            f.close()

        gen_data, seed = mdl.generate_data(100)
        gen_data = torch.cat([gen_data, -torch.ones(gen_data.size(0), 1, 320, 456).cuda()], dim=1)
        opts = {'title':'VAE generated data config {}'.format(config_num)}
        vis.images(0.5 + 0.5*gen_data.data.cpu(), opts=opts, win='VAE_config_{}'.format(config_num))

        pdb.set_trace()
        
        if compute_kdes:
            num_samples = 10
            av_lls, im_gen, im_test = compute_nparam_density(test_loader, mdl, 0.2, arguments.cuda, num_samples=num_samples)
            av_mmds = get_mmds(test_loader, mdl, 10, arguments.cuda, 
                               num_samples=num_samples)

            results.append((av_lls, Ks))
            mmds.append((av_mmds, Ks))

            vis.image(im_gen*0.5 + 0.5, win='VAE genim') 
            #vis.image(im_test*0.5 + 0.5, win='VAE testim') 


elif model == 'GAN':
    EP = 25

    Kss = [[100]]
    for config_num, Ks in enumerate(Kss):
        generator = netg_dcgan(Ks[0])
        discriminator = netd_dcgan()

        generator.weight_init(mean=0.0, std=0.02)
        discriminator.weight_init(mean=0.0, std=0.02)

        if arguments.cuda:
            generator.cuda()
            discriminator.cuda()

        path = 'models/GAN_{}_K_{}.t'.format(arguments.data, Ks)
        if os.path.exists(path):
            generator.load_state_dict(torch.load(path))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            adversarial__trainer(EP, train_loader, generator, discriminator, arguments, config_num, vis=vis)
            torch.save(generator.state_dict(), path)
        
        if compute_kdes:
            num_samples=10
            av_lls, im_gen, im_test = compute_nparam_density(test_loader, generator, 0.2, arguments.cuda, num_samples=num_samples)
            av_mmds = get_mmds(test_loader, generator, 10, arguments.cuda, 
                               num_samples=num_samples)

            results.append((av_lls, Ks))
            mmds.append((av_mmds, Ks))
elif model == 'GAN_W':
    EP = 250

    Kss = [[100]]
    for config_num, Ks in enumerate(Kss):
        generator = DCGAN_G(64, 100, 3, 128, ngpu=arguments.num_gpus, 
                            n_extra_layers=0)
        netg_dcgan_par(Ks[0])
        #discriminator = netd_dcgan_par()

        #generator.apply(weights_init_seq)
        #discriminator.apply(weights_init_seq)

        if arguments.cuda:
            generator.cuda()
            #discriminator.cuda()

        path = os.path.join(os.path.expanduser('~'), 'your_path', 'GANs', 'implicit_models',
                            'WassersteinGAN', 'GANW_128_netG_celeba_epoch_24.t')
        if 1: #& os.path.exists(path):
            generator.load_state_dict(torch.load(path))
            pdb.set_trace()
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            adversarial_wasserstein_trainer(train_loader, generator, discriminator, arguments=arguments, config_num=config_num, vis=vis, EP = EP)
            torch.save(generator.state_dict(), path)
        
        if compute_kdes:
            num_samples = 10
            av_lls, im_gen, im_test = compute_nparam_density(test_loader, generator, 0.2, arguments.cuda, num_samples=num_samples)
            av_mmds = get_mmds(test_loader, generator, 10, arguments.cuda, 
                               num_samples=num_samples)

            results.append((av_lls, Ks))
            mmds.append((av_mmds, Ks))

imagepath = os.path.expanduser('~')
imagepath = os.path.join(imagepath, 'your_path', 'GANs', 'implicit_models')
if not os.path.exists(imagepath):
    os.mkdir(imagepath)

N = 500 
if model == 'NF':
    gen_data, _ = mdl.generate_data(N=N, base_dist=base_dist)
elif model == 'VAE':
    gen_data, _ = mdl.generate_data(N)
elif model in ['GAN', 'GAN_W']:
    gen_data, _ = generator.generate_data(N=N)
gen_randoms = ut.collate_images_rectangular(gen_data.data, N=N, ncols=4, L1=456, L2=320 )

torch.save(gen_data, '/your_root/data2/mice_gen_randoms.t')
pdb.set_trace()
if 0:
    
    if 1:
        plt.figure(figsize=(8, 32), dpi=400)

        plt.imshow(0.5*gen_randoms.numpy()+0.5, interpolation='None')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(imagepath + '/more_randoms_{}_{}.png'.format(arguments.data, model), format='png')

    if compute_kdes:
        print(results)
        if 1:
            if model == 'NF':
                pickle.dump(im_test.numpy(), open(imagepath + '/test_images_{}.pickle'.format(arguments.data), 'wb')) 
            pickle.dump(im_gen.numpy(), open(imagepath + '/celeba_kde_images_{}.pickle'.format(model), 'wb'))
            pickle.dump(gen_randoms.numpy(), open(imagepath + '/celeba_random_{}.pickle'.format(model), 'wb'))
            pickle.dump(results, open(imagepath + '/celeba_Kvskde_{}.pickle'.format(model), 'wb')) 
if 0:
    print(results)
    print(mmds)
    
    pickle.dump(results, open(imagepath + '/celeba_results_thesis_am_{}.pickle'.format(model), 'wb')) 
    pickle.dump(mmds, open(imagepath + '/celeba_mmds_thesis_am_{}.pickle'.format(model), 'wb')) 




