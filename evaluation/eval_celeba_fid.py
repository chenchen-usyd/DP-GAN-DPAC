import os
import sys
import argparse
import numpy as np
import logging
import tensorflow as tf
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import frechet_inception_distance as FID

sys.path.insert(0, './../source')
from utils_celeba_gender import mkdir
from CelebA import MyCelebA

Img_W = Img_H = 32
Img_C = 3
DATA_ROOT = './../data'
STAT_DIR = './stats'


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-data', type=str, default='celeba', choices=['celeba'],
                        help=' dataset name')
    parser.add_argument('--gen_data', type=str,
                        default='./../results/celeba_gender/main/ResNet_default/gen_data.npz',
                        help='path of file that store the generated data')
    parser.add_argument('--save_dir', type=str,
                        help='output folder name; will be automatically save to the folder of gen_data if not specified')
    parser.add_argument('--num_eval_samples', type=int, default=20000,
                        help="number of samples to be evaluated")
    args = parser.parse_args()
    return args


##########################################################################
### helper functions
##########################################################################
def convert_data(data, Img_W, Img_H, Img_C):
    shape = data.shape
    if len(shape) == 2:
        data = np.reshape(data, [-1, Img_W, Img_H, Img_C])
    elif len(shape) == 3:
        data = np.reshape(data, [-1, Img_W, Img_H, Img_C])
    elif len(shape) == 4:
        if shape[1] == Img_C:
            data = np.transpose(data, [0, 2, 3, 1])

    if Img_C == 1:
        data = np.tile(data, [1, 1, 1, 3])
    return data * 255

def inf_train_gen(trainloader):
    while True:
        for images, targets in trainloader:
            yield (images, targets)

def load_celeba(selection='train_val'):
    transform_train_celeba = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    dataloader = MyCelebA
    
    if selection == 'train':
        trainset = dataloader(root=DATA_ROOT, split='train', download=False, transform=transform_train_celeba)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), drop_last=False, shuffle=False)
        input_data = inf_train_gen(trainloader)
        real_data, real_y = next(input_data)
        x = real_data.cpu().detach().numpy()
        y = real_y.cpu().detach().numpy()
    
    elif selection == 'train_val':
        trainset = dataloader(root=DATA_ROOT, split='train', download=False, transform=transform_train_celeba)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), drop_last=False, shuffle=False)
        input_data_train = inf_train_gen(trainloader)
        real_data_train, real_y_train = next(input_data_train)
        x_train = real_data_train.cpu().detach().numpy()
        y_train = real_y_train.cpu().detach().numpy()        

        valset = dataloader(root=DATA_ROOT, split='valid', download=False, transform=transform_train_celeba)
        valloader = torch.utils.data.DataLoader(valset, batch_size=len(valset), drop_last=False, shuffle=False)
        input_data_val = inf_train_gen(valloader)
        real_data_val, real_y_val = next(input_data_val)
        x_val = real_data_val.cpu().detach().numpy()
        y_val = real_y_val.cpu().detach().numpy()        
        
        x = np.concatenate([x_train, x_val])
        y = np.concatenate([y_train, y_val])
    
    elif selection == 'test':
        testset = dataloader(root=DATA_ROOT, split='test', download=False, transform=transform_train_celeba)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), drop_last=False, shuffle=False)
        input_data = inf_train_gen(testloader)
        real_data, real_y = next(input_data)
        x = real_data.cpu().detach().numpy()
        y = real_y.cpu().detach().numpy()
    
    elif selection == 'val':
        valset = dataloader(root=DATA_ROOT, split='valid', download=False, transform=transform_train_celeba)
        valloader = torch.utils.data.DataLoader(valset, batch_size=len(valset), drop_last=False, shuffle=False)
        input_data = inf_train_gen(valloader)
        real_data, real_y = next(input_data)
        x = real_data.cpu().detach().numpy()
        y = real_y.cpu().detach().numpy()
    
    print('data shape', x.shape, y.shape)
    print('data range:', np.min(x), np.max(x))
    return x, y

##########################################################################
### main
##########################################################################
def main(args):
    ### Get real data statistics
    stat_file = os.path.join(STAT_DIR, args.dataset, 'stat.npz')
    if not os.path.exists(stat_file):
        if args.dataset == 'celeba':
            real_data, _ = load_celeba('train_val')
        real_data = convert_data(real_data, Img_W, Img_H, Img_C)
        m1, s1, real_act = FID.calculate_activation(real_data)

        ## Save real statistics
        mkdir(os.path.join(STAT_DIR, args.dataset))
        np.savez(stat_file, mu=m1, sigma=s1, real_act=real_act)
    else:
        ## Load pre-computed statistics
        f = np.load(stat_file)
        m1, s1 = f['mu'][:], f['sigma'][:]
        real_act = f['real_act']
    print(m1.shape)
    print(s1.shape)
    print(real_act.shape)

    ### load gen data
    gen_data = np.load(args.gen_data)
    x_gen = gen_data['data_x']
    x_gen = convert_data(x_gen, Img_W, Img_H, Img_C)
    rand_perm = np.random.permutation(len(x_gen))
    x_gen = x_gen[rand_perm]
    x_gen = x_gen[:args.num_eval_samples]
    print(x_gen.shape)
    print(np.min(x_gen), np.max(x_gen))

    ### Get fake data statistics and compute FID
    m2, s2, fake_act = FID.calculate_activation(x_gen)
    fid_value = FID.calculate_frechet_distance(m1, s1, m2, s2)
    infostr = 'fid value: {}'.format(fid_value)
    print(infostr)



if __name__ == '__main__':
    args = parse_arguments()
    main(args)
