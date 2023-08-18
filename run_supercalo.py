# pylint: disable=invalid-name
""" Main script to run SuperCalo for Dataset 2 of the CaloChallenge

    by Ian Pang, John Andrew Raine and David Shih

"""

######################################   Imports   ################################################

import argparse
import os
import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from nflows import transforms, distributions, flows
import h5py
from nflows.utils import torchutils
from data_supercalo import get_calo_dataloader
from data_supercalo import get_coarse_voxels
from data_supercalo import get_flow2_data
torch.set_default_dtype(torch.float64)


#####################################   Parser setup   ############################################
parser = argparse.ArgumentParser()

parser.add_argument('--which_ds', default='2',
                    help='Which dataset to use: "2", "3" ')

# which flow uses bit flags: flow-I counts 1, flow-II counts 2, flow-III (i.e. SuperCalo A) counts 4.
# sum up which ones you want to work with
parser.add_argument('--which_flow', type=int, default=7,
                    help='Which flow(s) to train/generate. Default 7(=1+2+3).')
parser.add_argument('--train', action='store_true', help='train the setup')
parser.add_argument('--generate', action='store_true',
                    help='generate from a trained flow')
parser.add_argument('--upsample',  action='store_true',
                    help='upsample fine voxels from true coarse voxels')

parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

parser.add_argument('--output_dir', default='./results', help='Where to store the output')
parser.add_argument('--results_file', default='results.txt',
                    help='Filename where to store settings and test results.')

parser.add_argument('--data_dir', help='Where to find the training dataset')

parser.add_argument('--log_interval', type=int, default=175,
                    help='How often to show loss statistics and save samples.')

parser.add_argument('--noise_level', type=float, default=5e-3,
                    help='What level of noise to add to training data. Default is 5e-3')
parser.add_argument('--threshold_cut', type=float, default=1.5e-2,
                    help='What cut to apply after generation. Default is 1.5e-2')

# MAF parameters
parser.add_argument('--n_blocks', type=int, default='8',
                    help='Total number of blocks to stack in a model (MADE in MAF).')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size of flow training. Defaults to 1000.')
parser.add_argument('--hidden_size', type=int, default='128',
                    help='Hidden layer size for each MADE block in an MAF.')
parser.add_argument('--n_hidden', type=int, default=1,
                    help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu',
                    help='What activation function of torch.nn.functional to use in the MADEs.')
parser.add_argument('--n_bins', type=int, default=8,
                    help='Number of bins if piecewise transforms are used')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.,
                    help='dropout probability, defaults to 0')


#######################################   helper functions   ######################################

ALPHA = 1e-6
def logit(x):
    """ returns logit of input """
    return torch.log(x / (1.0 - x))

def sigmoid(x):
    """ returns sigmoid of input """
    return torch.exp(x) / (torch.exp(x) + 1.)

def logit_trafo(x):
    """ implements logit trafo of MAF paper https://arxiv.org/pdf/1705.07057.pdf """
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def inverse_logit(x, clamp_low=0., clamp_high=1.):
    """ inverts logit_trafo(), clips result if needed """
    return ((sigmoid(x) - ALPHA) / (1. - 2.*ALPHA)).clamp(clamp_low, clamp_high)

def add_noise(input_array, noise_level=1e-4):
    """ adds a bit of noise """
    noise = (torch.rand(size=input_array.size())*noise_level).to(input_array.device)
    return input_array+noise
    #return (input_array+noise)/(1.+noise_level)

def save_flow(model, number, arg):
    """ saves model to file """
    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(arg.output_dir, 'ds_{}_flow_{}.pt'.format(arg.which_ds, number)))
    print("Model saved")

def load_flow(model, number, arg):
    """ loads model from file """
    checkpoint = torch.load(os.path.join(arg.output_dir,
                                         'ds_{}_flow_{}.pt'.format(arg.which_ds, number)),
                            map_location=arg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(arg.device)
    model.eval()
    return model

def pad_front(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

    pad_size = target_length - array.shape[axis]
    
    if pad_size <= 0:
        return array
    
    npad = [(0, 0)] * array.ndim
    npad[axis] = (pad_size, 0)
    
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def pad_back(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

    pad_size = target_length - array.shape[axis]
    
    if pad_size <= 0:
        return array
    
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

class GuidedCompositeTransform(transforms.CompositeTransform):
    """Composes several transforms into one (in the order they are given),
       optionally returns intermediate results (steps) and NN outputs (p)"""

    def __init__(self, transforms):
        """Constructor.
        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__(transforms)
        self._transforms = torch.nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context, direction, return_steps=False, return_p=False):
        steps = [inputs]
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        ret_p = []
        for func in funcs:
            if hasattr(func.__self__, '_transform') and return_p:
                # in student IAF
                if direction == 'forward':
                    outputs, logabsdet = func(outputs, context)
                    ret_p.append(func.__self__._transform.autoregressive_net(outputs, context))
                else:
                    ret_p.append(func.__self__._transform.autoregressive_net(outputs, context))
                    outputs, logabsdet = func(outputs, context)
            elif hasattr(func.__self__, 'autoregressive_net') and return_p:
                # in teacher MAF
                if direction == 'forward':
                    ret_p.append(func.__self__.autoregressive_net(outputs, context))
                    outputs, logabsdet = func(outputs, context)
                else:
                    outputs, logabsdet = func(outputs, context)
                    ret_p.append(func.__self__.autoregressive_net(outputs, context))
            else:
                outputs, logabsdet = func(outputs, context)
            steps.append(outputs)
            total_logabsdet += logabsdet
        if return_steps and return_p:
            return outputs, total_logabsdet, steps, ret_p
        elif return_steps:
            return outputs, total_logabsdet, steps
        elif return_p:
            return outputs, total_logabsdet, ret_p
        else:
            return outputs, total_logabsdet

    def forward(self, inputs, context=None, return_steps=False, return_p=False):
        #funcs = self._transforms
        funcs = (transform.forward for transform in self._transforms)
        return self._cascade(inputs, funcs, context, direction='forward',
                             return_steps=return_steps, return_p=return_p)

    def inverse(self, inputs, context=None, return_steps=False, return_p=False):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context, direction='inverse',
                             return_steps=return_steps, return_p=return_p)

def chi2_loss(input1, input2):
    ret = (((input1 - input2)**2).sum(dim=1)).mean()
    return ret

def logabsdet_of_base(noise, width=1.):
    """ for computing KL of student"""
    shape = noise.size()[1]
    ret = -0.5 * torchutils.sum_except_batch((noise/width) ** 2, num_batch_dims=1)
    log_z = torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64)
    return ret - log_z

##############################################     build flows      ########################################################

def build_flow_1(features, context_features, arg, hidden_size, num_layers=1):
    """ returns flow-I and optimizer """
    flow_params_RQS = {'num_blocks': num_layers, # num of hidden layers per block
                       'use_residual_blocks': False,
                       'use_batch_norm': False,
                       'dropout_probability': 0,
                       'activation': F.relu,
                       'random_mask':False,
                       'num_bins': arg.n_bins,
                       'tails':'linear',
                       'tail_bound': 14.,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}
    flow_blocks = []
    for i in range(arg.n_blocks):
        flow_blocks.append(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            **flow_params_RQS,
            features=features,
            context_features=context_features,
            hidden_features=hidden_size))
        if i%2 == 0:
            flow_blocks.append(transforms.ReversePermutation(features))
        else:
            flow_blocks.append(transforms.RandomPermutation(features))

    del flow_blocks[-1]
    flow_transform = GuidedCompositeTransform(flow_blocks)
    flow_base_distribution = distributions.StandardNormal(shape=[features])

    flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)

    model = flow.to(arg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1E-4, total_steps=35000, epochs=500, steps_per_epoch=None, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=10.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1, verbose=False)
    print(model)
    print(model, file=open(arg.results_file, 'a'))


    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Flow-I has {} parameters".format(total_parameters))
    print("Flow-I has {} parameters".format(total_parameters), file=open(arg.results_file, 'a'))
    return model, optimizer, lr_schedule

def build_flow_2(features, context_features, arg, hidden_size, num_layers=1):
    """ returns flow-II and optimizer """
    flow_params_RQS = {'num_blocks': num_layers, # num of hidden layers per block
                       'use_residual_blocks': False,
                       'use_batch_norm': False,
                       'dropout_probability': arg.dropout_probability,
                       'activation': F.relu,
                       'random_mask':False,
                       'num_bins': arg.n_bins,
                       'tails':'linear',
                       'tail_bound': 6.,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}
    flow_blocks = []
    for i in range(arg.n_blocks):
        flow_blocks.append(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            **flow_params_RQS,
            features=features,
            context_features=context_features,
            hidden_features=hidden_size))
        if i%2 == 0:
            flow_blocks.append(transforms.ReversePermutation(features))
        else:
            flow_blocks.append(transforms.RandomPermutation(features))

    del flow_blocks[-1]
    flow_transform = GuidedCompositeTransform(flow_blocks)
    flow_base_distribution = distributions.StandardNormal(shape=[features])

    flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)

    model = flow.to(arg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)


    lr_schedule = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                       base_lr=0.5E-4, max_lr=2.0E-3,
                                                       step_size_up=350, step_size_down=None,
                                                       mode='triangular2', gamma=0.8, scale_fn=None, scale_mode='cycle',
                                                       cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1,
                                                           verbose=False)
 
    print(model)
    print(model, file=open(arg.results_file, 'a'))


    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Flow-II has {} parameters".format(total_parameters))
    print("Flow-II has {} parameters".format(total_parameters), file=open(arg.results_file, 'a'))
    return model, optimizer, lr_schedule


def build_flow_3(features, context_features, arg, hidden_size, num_layers=1):
    """ returns build SuperCalo A and optimizer """
    flow_params_RQS = {'num_blocks': num_layers, # num of hidden layers per block
                       'use_residual_blocks': False,
                       'use_batch_norm': False,
                       'dropout_probability': arg.dropout_probability,
                       'activation': F.relu,
                       'random_mask':False,
                       'num_bins': arg.n_bins,
                       'tails':'linear',
                       'tail_bound': 14.,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}
    flow_blocks = []
    for i in range(arg.n_blocks):
        flow_blocks.append(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            **flow_params_RQS,
            features=features,
            context_features=context_features,
            hidden_features=hidden_size))
        if i%2 == 0:
            flow_blocks.append(transforms.ReversePermutation(features))
        else:
            flow_blocks.append(transforms.RandomPermutation(features))

    del flow_blocks[-1]
    flow_transform = GuidedCompositeTransform(flow_blocks)
    flow_base_distribution = distributions.StandardNormal(shape=[features])

   # embedding_net=EmbeddingLayer()
    flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)

    model = flow.to(arg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1E-3, total_steps=30240, epochs=40, steps_per_epoch=None, pct_start=0.45, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=50.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1, verbose=False)
    print(model)
    print(model, file=open(arg.results_file, 'a'))


    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("SuperCalo A has {} parameters".format(total_parameters))
    print("SuperCalo A has {} parameters".format(total_parameters), file=open(arg.results_file, 'a'))
    return model, optimizer, lr_schedule

##############################################     generate from flows      ########################################################

@torch.no_grad()
def generate_flow_1(flow, arg, num_samples, energies=None):
    """ samples from flow 1 and returns E_i and E_inc in MeV """
    if energies is None:
        energies = (torch.rand(size=(num_samples, 1))*3. - 1.5).to(arg.device)
    samples = flow.sample(1, energies).reshape(len(energies), -1)
    samples = inverse_logit(samples) * arg.normalization
    samples = torch.where(samples < arg.threshold_cut, torch.zeros_like(samples), samples)
    return 10**(energies + 4.5), samples

@torch.no_grad()
def generate_flow_2(flow, arg, e_inc, e_layer):
    cond_inc = torch.log10(e_inc)-4.5
    e_layer = add_noise(e_layer, noise_level=0.36)
    cond_Edep = torch.log10(e_layer)/4
    cond = torch.vstack([cond_inc.T, cond_Edep.T]).T.to(arg.device)
    coarse = flow.sample(1, cond).reshape(len(cond), -1)
    coarse = 10**(coarse-6)
    coarse = 28048*coarse
    coarse = torch.where(coarse < arg.threshold_cut, torch.zeros_like(coarse), coarse)
    return coarse

@torch.no_grad()
def generate_flow_3(flow, arg, e_inc, e_layer, e_coarse):
    e_coarse = add_noise(e_coarse, noise_level = 5e-3)

    e_coarse_final = torch.reshape(e_coarse, (-1,1))
    e_coarse = torch.reshape(e_coarse, (-1,9,9,8))
    cond_coarse = logit_trafo(e_coarse_final.to(arg.device)/28048)

    coarse_adj_alpha_1 = torch.roll(e_coarse,1,dims=3)
    coarse_adj_alpha_1= torch.reshape(coarse_adj_alpha_1, (-1,1))
    coarse_adj_alpha_2 = torch.roll(e_coarse,-1,dims=3)
    coarse_adj_alpha_2= torch.reshape(coarse_adj_alpha_2, (-1,1))
    coarse_adj_r_1 = e_coarse[:,:,:-1,:] #adjacent inner coarse voxel in r
    coarse_adj_r_1 = pad_front(coarse_adj_r_1.cpu().numpy(),9,axis=2) #pad zeros for inner most coarse voxels
    coarse_adj_r_1 = torch.reshape(torch.from_numpy(coarse_adj_r_1), (-1,1))
    coarse_adj_r_2 = e_coarse[:,:,1:,:] #adjacent outer coarse voxel in r  
    coarse_adj_r_2 = pad_back(coarse_adj_r_2.cpu().numpy(),9,axis=2) #pad zeros for outer most coarse voxels
    coarse_adj_r_2 = torch.reshape(torch.from_numpy(coarse_adj_r_2), (-1,1))
    coarse_adj_z_1 = e_coarse[:,:-1,:,:]
    coarse_adj_z_1 = pad_front(coarse_adj_z_1.cpu().numpy(),9,axis=1)
    coarse_adj_z_1 = torch.reshape(torch.from_numpy(coarse_adj_z_1), (-1,1))
    coarse_adj_z_2 = e_coarse[:,1:,:,:]
    coarse_adj_z_2 = pad_back(coarse_adj_z_2.cpu().numpy(),9,axis=1)
    coarse_adj_z_2 = torch.reshape(torch.from_numpy(coarse_adj_z_2), (-1,1))


    e_layer = torch.reshape(e_layer, (-1,9,5))
    e_layer =  torch.repeat_interleave(e_layer, int(16*9/2),dim=1)
    e_layer = torch.reshape(e_layer, (-1,5))

    cond_Edep = logit_trafo(e_layer.to(arg.device)/arg.normalization)
    cond_adj_alpha_1 = logit_trafo(coarse_adj_alpha_1.to(arg.device)/28048)
    cond_adj_alpha_2 = logit_trafo(coarse_adj_alpha_2.to(arg.device)/28048)
    cond_adj_r_1 = logit_trafo(coarse_adj_r_1.to(arg.device)/28048)
    cond_adj_r_2 = logit_trafo(coarse_adj_r_2.to(arg.device)/28048)
    cond_adj_z_1 = logit_trafo(coarse_adj_z_1.to(arg.device)/28048)
    cond_adj_z_2 = logit_trafo(coarse_adj_z_2.to(arg.device)/28048)

    cond_z = []
    for i in range(9):
        cond_z += list(i*torch.ones(72))
    cond_z = np.tile(cond_z,len(e_inc))

    cond_radial = []
    for i in range(9):
        for j in range(9):
            cond_radial += list(j*torch.ones(8))
    cond_radial = np.tile(cond_radial,len(e_inc))

    cond_radial = F.one_hot(torch.from_numpy(cond_radial).long(),num_classes = 9).to(args.device)
    cond_z =  F.one_hot(torch.from_numpy(cond_z).long(),num_classes = 9).to(args.device)

    e_inc = torch.repeat_interleave(e_inc, int(45*16*9/10))
    cond_inc = torch.unsqueeze(torch.log10(e_inc.to(arg.device))-4.5,1)
    cond = torch.vstack([cond_inc.T, cond_Edep.T, cond_coarse.T, cond_adj_r_1.T,cond_adj_r_2.T, cond_adj_z_1.T,cond_adj_z_2.T,cond_adj_alpha_1.T,cond_adj_alpha_2.T, cond_radial.T,cond_z.T]).T

    with torch.no_grad():
        fine = flow.sample(1, cond).reshape(len(cond), -1)
        fine_vox = inverse_logit(fine)
        fine_vox = fine_vox/fine_vox.sum(dim=-1, keepdims=True)*e_coarse_final 
        fine_vox = torch.where(fine_vox < args.threshold_cut, torch.zeros_like(fine_vox), fine_vox) 
  
    return fine_vox

##############################################     train and evaluate flows      ########################################################
def train_eval_flow_1(flow, optimizer, schedule, train_loader, test_loader, arg):
    """ train flow 1, learning p(E_i|E_inc), eval after each epoch """

    num_epochs = 500                                                                                                                                                                                                                                                     
    best_LL = -np.inf
    for epoch in range(num_epochs):
        # train:                                                                                                                                                                                                                                                              
        for idx, batch in enumerate(train_loader):
            flow.train()
            e_dep = batch['energy_dep'].to(arg.device)
            e_dep = logit_trafo(e_dep/arg.normalization)
            cond = torch.log10(batch['energy_inc'].to(arg.device))-4.5
            loss = - flow.log_prob(e_dep, cond).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            if idx % 10 == 0:
                    print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                        epoch+1, num_epochs, idx+1, len(train_loader), loss.item()))
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                        epoch+1, num_epochs, idx+1, len(train_loader), loss.item()),
                        file=open(arg.results_file, 'a'))

        logprb_mean, logprb_std = eval_flow_1(test_loader, flow, arg)

        output = 'Evaluate (epoch {}) -- '.format(epoch+1) +\
            'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
        print(output.format(logprb_mean, logprb_std))
        print(output.format(logprb_mean, logprb_std),
              file=open(arg.results_file, 'a'))
        if logprb_mean > best_LL:
            best_LL = logprb_mean
            save_flow(flow, 1, arg)
        #schedule.step()                                                                                                                                                                                                                                                      
    flow = load_flow(flow, 1, arg)

@torch.no_grad()
def eval_flow_1(test_loader, flow, arg):
    """ returns LL of data in dataloader for flow 1"""
    loglike = []
    flow.eval()
    for _, batch in enumerate(test_loader):
        e_dep = batch['energy_dep'].to(arg.device)
        e_dep = logit_trafo(e_dep/arg.normalization)
        cond = torch.log10(batch['energy_inc'].to(arg.device))-4.5

        loglike.append(flow.log_prob(e_dep, cond))

    logprobs = torch.cat(loglike, dim=0)

    logprb_mean = logprobs.mean(0)
    logprb_std = logprobs.var(0).sqrt()

    return logprb_mean, logprb_std


def train_eval_flow_2(flow, optimizer, schedule, train_loader, test_loader, arg):
    num_epochs = 40

    best_LL = -np.inf

    for epoch in range(num_epochs):
        # train:
        for idx, batch in enumerate(train_loader):
            flow.train()
            cond_inc = torch.log10(batch['energy'].to(arg.device))-4.5
            cond_Edep = torch.log10(batch['energy_dep'].to(arg.device))/4
            coarse_voxels = torch.log10(batch['coarse_voxels'].to(arg.device))+6
            cond = torch.vstack([cond_inc.T, cond_Edep.T]).T

            loss = - flow.log_prob(coarse_voxels, cond).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            if idx % 189 == 0:
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()))
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()),
                      file=open(arg.results_file, 'a'))

        logprb_mean, logprb_std = eval_flow_2(test_loader, flow, arg)

        output = 'Evaluate (epoch {}) -- '.format(epoch+1) +\
            'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
        print(output.format(logprb_mean, logprb_std))
        print(output.format(logprb_mean, logprb_std),
              file=open(arg.results_file, 'a'))
        if logprb_mean > best_LL:
            best_LL = logprb_mean
            save_flow(flow, 2, arg)
    flow = load_flow(flow, 2, arg)
    arg.best_LL = best_LL

@torch.no_grad()
def eval_flow_2(test_loader, flow, arg):
    """ returns LL of data in dataloader for flow 3"""
    loglike = []
    flow.eval()
    for _, batch in enumerate(test_loader):
        cond_inc = torch.log10(batch['energy'].to(arg.device))-4.5
        cond_Edep = torch.log10(batch['energy_dep'].to(arg.device))/4#/299347)
        coarse_voxels = torch.log10(batch['coarse_voxels'].to(arg.device))+6
        cond = torch.vstack([cond_inc.T, cond_Edep.T]).T

        loglike.append(flow.log_prob(coarse_voxels, cond))

    logprobs = torch.cat(loglike, dim=0)

    logprb_mean = logprobs.mean(0)
    logprb_std = logprobs.var(0).sqrt()

    return logprb_mean, logprb_std


def train_eval_flow_3(flow, optimizer, schedule, train_loader, test_loader, arg):
    """ train flow 3, learning p(I_n|I_(n-1), E_n, E_(n-1), E_inc) eval after each epoch"""

    num_epochs = 40 
    if vars(arg).get('best_LL') is None:
        best_LL = -np.inf
    else:
        best_LL = arg.best_LL
    for epoch in range(num_epochs):
        # train:
        for idx, batch in enumerate(train_loader):
            flow.train()
            cond_inc = torch.unsqueeze(torch.log10(batch['energy_inc'].to(arg.device))-4.5,1)
            cond_coarse = torch.unsqueeze(logit_trafo(batch['coarse_voxels'].to(arg.device)/28048),1)
            cond_Edep = logit_trafo(batch['energy_dep'].to(arg.device)/arg.normalization)
            #################### conditioning on all neighbors ######################################
            cond_adj_alpha_1 = torch.unsqueeze(logit_trafo(batch['E_adj_alpha_1'].to(arg.device)/28048),1)
            cond_adj_alpha_2 = torch.unsqueeze(logit_trafo(batch['E_adj_alpha_2'].to(arg.device)/28048),1)
            cond_adj_r_1 = torch.unsqueeze(logit_trafo(batch['E_adj_r_1'].to(arg.device)/28048),1)
            cond_adj_r_2 = torch.unsqueeze(logit_trafo(batch['E_adj_r_2'].to(arg.device)/28048),1)
            cond_adj_z_1 = torch.unsqueeze(logit_trafo(batch['E_adj_z_1'].to(arg.device)/28048),1)
            cond_adj_z_2 = torch.unsqueeze(logit_trafo(batch['E_adj_z_2'].to(arg.device)/28048),1)
            ###########################################################################################

            fine_voxels = logit_trafo(batch['fine_voxels'].to(arg.device)/(torch.unsqueeze(batch['coarse_voxels'].to(arg.device),1)))

            cond_radial = F.one_hot(batch['r_labels'].long(),num_classes = 9).to(arg.device)
            cond_z =  F.one_hot(batch['z_labels'].long(),num_classes = 9).to(arg.device)

            cond = torch.vstack([cond_inc.T, cond_Edep.T, cond_coarse.T, cond_adj_r_1.T,cond_adj_r_2.T, cond_adj_z_1.T,cond_adj_z_2.T,cond_adj_alpha_1.T,cond_adj_alpha_2.T, cond_radial.T,cond_z.T]).T

            loss = - flow.log_prob(fine_voxels, cond).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            if idx % 189 == 0:
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()))
                print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, idx+1, len(train_loader), loss.item()),
                      file=open(arg.results_file, 'a'))

        logprb_mean, logprb_std = eval_flow_3(test_loader, flow, arg)

        output = 'Evaluate (epoch {}) -- '.format(epoch+1) +\
            'logp(x, at E(x)) = {:.3f} +/- {:.3f}'
        print(output.format(logprb_mean, logprb_std))
        print(output.format(logprb_mean, logprb_std),
              file=open(arg.results_file, 'a'))
        if logprb_mean > best_LL:
            best_LL = logprb_mean
            save_flow(flow, 3, arg)
    flow = load_flow(flow, 3, arg)
    arg.best_LL = best_LL


@torch.no_grad()
def eval_flow_3(test_loader, flow, arg):
    """ returns LL of data in dataloader for flow 3"""
    loglike = []
    flow.eval()
    for _, batch in enumerate(test_loader):
        cond_inc = torch.unsqueeze(torch.log10(batch['energy_inc'].to(arg.device))-4.5,1)
        cond_coarse = torch.unsqueeze(logit_trafo(batch['coarse_voxels'].to(arg.device)/28048),1)
        cond_Edep = logit_trafo(batch['energy_dep'].to(arg.device)/arg.normalization)
        #################### conditioning on all neighbors ######################################
        cond_adj_alpha_1 = torch.unsqueeze(logit_trafo(batch['E_adj_alpha_1'].to(arg.device)/28048),1)
        cond_adj_alpha_2 = torch.unsqueeze(logit_trafo(batch['E_adj_alpha_2'].to(arg.device)/28048),1)
        cond_adj_r_1 = torch.unsqueeze(logit_trafo(batch['E_adj_r_1'].to(arg.device)/28048),1)
        cond_adj_r_2 = torch.unsqueeze(logit_trafo(batch['E_adj_r_2'].to(arg.device)/28048),1)
        cond_adj_z_1 = torch.unsqueeze(logit_trafo(batch['E_adj_z_1'].to(arg.device)/28048),1)
        cond_adj_z_2 = torch.unsqueeze(logit_trafo(batch['E_adj_z_2'].to(arg.device)/28048),1)
        ###########################################################################################

        fine_voxels = logit_trafo(batch['fine_voxels'].to(arg.device)/(torch.unsqueeze(batch['coarse_voxels'].to(arg.device),1)))

        cond_radial = F.one_hot(batch['r_labels'].long(),num_classes = 9).to(arg.device)
        cond_z =  F.one_hot(batch['z_labels'].long(),num_classes = 9).to(arg.device)

        cond = torch.vstack([cond_inc.T, cond_Edep.T, cond_coarse.T, cond_adj_r_1.T,cond_adj_r_2.T, cond_adj_z_1.T,cond_adj_z_2.T,cond_adj_alpha_1.T,cond_adj_alpha_2.T, cond_radial.T,cond_z.T]).T

        loglike.append(flow.log_prob(fine_voxels, cond))

    logprobs = torch.cat(loglike, dim=0)

    logprb_mean = logprobs.mean(0)
    logprb_std = logprobs.var(0).sqrt()

    return logprb_mean, logprb_std

##############################################     save to file      ########################################################


def save_to_file(incident, shower, arg, isflow2 = False):
    """ saves incident energies and showers to hdf5 file """
    if isflow2:
        filename = os.path.join(arg.output_dir, 'supercalo_ds_2_coarse.hdf5')
    else:
        filename = os.path.join(arg.output_dir, 'supercalo_ds_2_full.hdf5')
    dataset_file = h5py.File(filename, 'w')
    dataset_file.create_dataset('incident_energies',
                                data=incident.reshape(len(incident), -1), compression='gzip')
    dataset_file.create_dataset('showers',
                                data=shower.reshape(len(shower), -1), compression='gzip')
    dataset_file.close()


###################################################################################################
#######################################   running the code   ######################################
###################################################################################################

if __name__ == '__main__':
    args = parser.parse_args()

    LAYER_SIZE = {'2': 9 * 16, '3': 18 * 50}[args.which_ds]
    DEPTH = 45
    args.normalization = 6.5e4

    # check if output_dir exists and 'move' results file there
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    args.results_file = os.path.join(args.output_dir, args.results_file)
    print(args, file=open(args.results_file, 'a'))

    # setup device
    args.device = torch.device('cuda:'+str(args.which_cuda) \
                               if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Using {}".format(args.device))
    print("Using {}".format(args.device), file=open(args.results_file, 'a'))

    preprocessing_kwargs = {'with_noise': True,
                            'noise_level': args.noise_level,
                            'apply_logit': True,
                            'do_normalization': True}

    if bin(args.which_flow)[-1] == '1':
        print("Working on Flow-I")
        print("Working on Flow-I", file=open(args.results_file, 'a'))
        if args.train:
                                                                                                                                                             
            train_loader_1, test_loader_1 = get_calo_dataloader(
                os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)),
                1, args.device,
                which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)                                                                                                                                                              

        flow_1, optimizer_1, schedule_1 = build_flow_1(DEPTH, 1, args, 256)

        if args.train:
            print("Training Flow-I ...")
            train_eval_flow_1(flow_1, optimizer_1, schedule_1, train_loader_1, test_loader_1, args)

        if args.generate:
            print("Generating from Flow-I ...")
            num_events = 10000
            num_batches = 10
            flow_1 = load_flow(flow_1, 1, args)
            incident_energies = []
            samples_1 = []
            for gen_batch in range(num_batches):
                incident_energies_loc, samples_1_loc = generate_flow_1(flow_1, args, num_events)
                np.save(os.path.join(args.output_dir, 'e_inc_1_{}.npy'.format(gen_batch)),
                        incident_energies_loc.cpu().numpy())
                np.save(os.path.join(args.output_dir, 'samples_1_{}.npy'.format(gen_batch)),
                        samples_1_loc.cpu().numpy())
                incident_energies.append(incident_energies_loc.cpu().numpy())
                samples_1.append(samples_1_loc.cpu().numpy())
                print("Done with generation batch {}/{}".format(gen_batch+1, num_batches))
                print("Done with generation batch {}/{}".format(gen_batch+1, num_batches),
                      file=open(args.results_file, 'a'))
            incident_energies = np.concatenate([*incident_energies])
            samples_1 = np.concatenate([*samples_1])
            np.save(os.path.join(args.output_dir, 'e_inc_1.npy'), incident_energies)
            np.save(os.path.join(args.output_dir, 'samples_1.npy'), samples_1)

    if bin(args.which_flow)[-2] == '1':
        print("Working on Flow 2")
        print("Working on Flow 2", file=open(args.results_file, 'a'))

        if args.train:
            train_loader_2, test_loader_2 = get_calo_dataloader(
                os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)),
                2, args.device,
                which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)

        flow_2, optimizer_2, schedule_2 = build_flow_2(648, 10, args, 648,
                                                    num_layers=1)

        if args.train:
            print("Training Flow-II ...")
            train_eval_flow_2(flow_2, optimizer_2, schedule_2, train_loader_2, test_loader_2, args)

        if args.generate:
            # generate coarse voxel energies from flow-II using true conditional info and save to file
            print("Generating coarse voxels using true conditional inputs...")
            flow_2 = load_flow(flow_2, 2, args)
            data = get_flow2_data(os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds))
            , args.device, which_ds=args.which_ds, batch_size=1000,
                    small_file=False, **preprocessing_kwargs)
            incident_energies = []
            samples = []
            for i, batch in enumerate(data):
                cond_inc = torch.log10(batch['energy'].to(args.device))-4.5
                cond_Edep = torch.log10(batch['energy_dep'].to(args.device))/4
                cond = torch.vstack([cond_inc.T, cond_Edep.T]).T
                with torch.no_grad():
                    coarse = flow_2.sample(1, cond).reshape(len(cond), -1)
                    coarse = 10**(coarse-6)

                incident_energies.append(batch['energy'].cpu().numpy())
                samples.append(coarse.cpu().numpy())

                print('Progress: {}% '.format((i+1)))
    
            incident_energies = np.reshape(incident_energies,(100000,1))

            samples = 28048*np.reshape(samples,(100000,648))
            samples = np.where(samples < args.threshold_cut, np.zeros_like(samples), samples)
            save_to_file(incident_energies, samples, args, isflow2=True)

    
    if bin(args.which_flow)[-3] == '1':
        print("Working on Flow 3")
        print("Working on Flow 3", file=open(args.results_file, 'a'))

        if args.train:
            train_loader_3, test_loader_3 = get_calo_dataloader(
                os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds)),
                3, args.device,
                which_ds=args.which_ds, batch_size=60000,
                small_file=(args.which_ds == '3'), **preprocessing_kwargs)
        flow_3, optimizer_3, schedule_3 = build_flow_3(10, 31, args,
                                                    args.hidden_size,
                                                        num_layers=2)

        if args.train:
            print("Training SuperCalo A ...")
            train_eval_flow_3(flow_3, optimizer_3, schedule_3, train_loader_3, test_loader_3, args)
            if args.which_ds == '3':
                del train_loader_3, test_loader_3
                train_loader_3, test_loader_3 = get_calo_dataloader(
                    os.path.join(args.data_dir, 'dataset_{}_2.hdf5'.format(args.which_ds)),
                    3, args.device, small_file=(args.which_ds == '3'),
                    which_ds=args.which_ds, batch_size=args.batch_size, **preprocessing_kwargs)
                train_eval_flow_3(flow_3, optimizer_3, schedule_3, train_loader_3, test_loader_3,
                                args)

        if args.upsample:
            # here the fine voxels are upsampled from the true coarse voxels from the training dataset
            print("Upsampling from SuperCalo using true coarse voxels ...")
            flow_3, optimizer_3, schedule_3 = build_flow_3(10, 31, args,
                                                    args.hidden_size,
                                                            num_layers=2)
            flow_3 = load_flow(flow_3, 3, args)
            coarse_data = get_coarse_voxels(os.path.join(args.data_dir, 'dataset_{}_1.hdf5'.format(args.which_ds))
            , args.device, which_ds=args.which_ds, batch_size=1000,
                    small_file=False, **preprocessing_kwargs)

            incident_energies = []
            course_voxels = []
            samples = []
            for i, batch in enumerate(coarse_data):
                cond_inc = torch.unsqueeze(torch.log10(batch['energy_inc'].to(args.device))-4.5,1)
                cond_coarse = torch.unsqueeze(logit_trafo(batch['coarse_voxels'].to(args.device)/28048),1)
                cond_Edep = logit_trafo(batch['energy_dep'].to(args.device)/args.normalization)                                                                                                                                                                           
                #################### conditioning on all neighbors ######################################                                                                                                                                                                         
                cond_adj_alpha_1 = torch.unsqueeze(logit_trafo(batch['E_adj_alpha_1'].to(args.device)/28048),1)
                cond_adj_alpha_2 = torch.unsqueeze(logit_trafo(batch['E_adj_alpha_2'].to(args.device)/28048),1)
                cond_adj_r_1 = torch.unsqueeze(logit_trafo(batch['E_adj_r_1'].to(args.device)/28048),1)
                cond_adj_r_2 = torch.unsqueeze(logit_trafo(batch['E_adj_r_2'].to(args.device)/28048),1)
                cond_adj_z_1 = torch.unsqueeze(logit_trafo(batch['E_adj_z_1'].to(args.device)/28048),1)
                cond_adj_z_2 = torch.unsqueeze(logit_trafo(batch['E_adj_z_2'].to(args.device)/28048),1)
                ###########################################################################################                                                                                                                                                                       

                cond_radial = F.one_hot(batch['r_labels'].long(),num_classes = 9).to(args.device)
                cond_z =  F.one_hot(batch['z_labels'].long(),num_classes = 9).to(args.device)

                cond = torch.vstack([cond_inc.T, cond_Edep.T, cond_coarse.T, cond_adj_r_1.T,cond_adj_r_2.T, cond_adj_z_1.T,cond_adj_z_2.T,cond_adj_alpha_1.T,cond_adj_alpha_2.T, cond_radial.T,cond_z.T]).T
                true_voxels = batch['fine_voxels']
                with torch.no_grad():
                    fine = flow_3.sample(1, cond).reshape(len(cond), -1)
                    fine_vox = inverse_logit(fine)
                    coarse_voxels = torch.unsqueeze(batch['coarse_voxels'].to(args.device),1)
                    coarse_voxels = torch.where(coarse_voxels < args.threshold_cut, torch.zeros_like(coarse_voxels), coarse_voxels)
                    fine_vox = fine_vox/fine_vox.sum(dim=-1, keepdims=True)*coarse_voxels 
                    fine_vox = torch.where(fine_vox < args.threshold_cut, torch.zeros_like(fine_vox), fine_vox) 
                incident_energies.append(batch['energy_inc'].cpu().numpy())
                samples.append(fine_vox.cpu().numpy())

                if (i+1)%648 == 0:
                    print('Progress: {}% '.format((i+1)/648))
    
            incident_energies = np.reshape(incident_energies,(64800000,1))
            incident_energies = incident_energies[647::648]
            incident_energies = np.reshape(incident_energies,(100000,1))

            samples = np.reshape(samples,(64800000,10))
            samples = np.reshape(samples,(100000,9,9,8,5,2))
            samples = np.swapaxes(samples,2,3)
            samples = np.swapaxes(samples,3,4)
            samples = np.swapaxes(samples,4,5)
            samples = np.swapaxes(samples,2,3)
            samples = np.reshape(samples,(100000,9,5,16,9))
            samples = np.reshape(samples,(100000,45,16,9))
            samples = np.reshape(samples,(100000,6480))
            save_to_file(incident_energies, samples, args)

        if args.generate:
            # here the fine voxels are generated using the full model chain (Flow-I + Flow-II + SuperCalo A)
            print("Generating from full model chain ...")
            num_events = 1000
            num_batches = 100

            full_start_time = time.time()
            flow_1, _, _ = build_flow_1(DEPTH, 1, args, 256)
            flow_1 = load_flow(flow_1, 1, args)
            flow_2, _, _ = build_flow_2(648, 10, args, 648, num_layers=1)
            flow_2 = load_flow(flow_2, 2, args)
            flow_3, optimizer_3, schedule_3 = build_flow_3(10, 31, args,
                                                    args.hidden_size,
                                                            num_layers=2)
            flow_3 = load_flow(flow_3, 3, args)
            incident_energies = []
            voxels = []

            i = 1
            start_time = time.time()
            for gen_batch in range(num_batches):
                incident_energies_loc, samples_1_loc = generate_flow_1(flow_1, args, num_events)

                incident_energies.append(incident_energies_loc.cpu().numpy())
                samples_1_loc_coarse = samples_1_loc.reshape(-1,9,5).sum(axis=-1)
                samples_2_loc = generate_flow_2(flow_2, args, incident_energies_loc, samples_1_loc_coarse)
                
                samples_3_loc = generate_flow_3(flow_3, args, incident_energies_loc, samples_1_loc, samples_2_loc)
                voxels.append(samples_3_loc.cpu().numpy())
                if i%10==0:
                    print('Progress: {}% '.format((i)))
                i += 1
            incident_energies = np.concatenate([*incident_energies])
            voxels = np.concatenate([*voxels])

            samples = np.reshape(voxels,(64800000,10))
            samples = np.reshape(samples,(100000,9,9,8,5,2))
            samples = np.swapaxes(samples,2,3)
            samples = np.swapaxes(samples,3,4)
            samples = np.swapaxes(samples,4,5)
            samples = np.swapaxes(samples,2,3)
            samples = np.reshape(samples,(100000,9,5,16,9))
            samples = np.reshape(samples,(100000,45,16,9))
            samples = np.reshape(samples,(100000,6480))
            save_to_file(incident_energies, samples, args)
            end_time = time.time()
            total_time = end_time-start_time
            time_string = "Full chain: Needed {:d} min and {:.1f} s to generate {} events in {} batch(es)."+" This means {:.2f} ms per event."
            print(time_string.format(int(total_time//60), total_time%60, 100000,
                                        num_batches, total_time*1e3 /100000))
    print("DONE with everything!")
    print("DONE with everything!", file=open(args.results_file, 'a'))
