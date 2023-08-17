# pylint: disable=invalid-name
""" Dataloader for calorimeter data used in SuperCalo study

    by Ian Pang, John Andrew Raine and David Shih

"""

import os

import numpy as np
import h5py
import torch

from torch.utils.data import Dataset, DataLoader

def add_noise(input_array, noise_level=1e-4):
    noise = np.random.rand(*input_array.shape)*noise_level
    return input_array+noise

ALPHA = 1e-6
def logit(x):
    """ returns logit of input """
    return np.log(x / (1.0 - x))

def sigmoid(x):
    """ returns sigmoid of input """
    return np.exp(x) / (np.exp(x) + 1.)

def logit_trafo(x):
    """ implements logit trafo of MAF paper https://arxiv.org/pdf/1705.07057.pdf """
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def inverse_logit(x, clamp_low=0., clamp_high=1.):
    """ inverts logit_trafo(), clips result if needed """
    return ((sigmoid(x) - ALPHA) / (1. - 2.*ALPHA)).clip(clamp_low, clamp_high)

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

class CaloDataLayerEnergy(Dataset):
    """ Dataloader for E_i of each layer (flow-1)"""
    def __init__(self, path_to_file, which_ds='2',
                 beginning_idx=0, data_length=100000,
                 **preprocessing_kwargs):
        """
        Args:
            path_to_file (string): path to .hdf5 file
            which_ds ('2' or '3'): which dataset (kind of redundant with path_to_file name)
            beginning_idx (int): at which index to start taking the data from
            data_length (int): how many events to take
            preprocessing_kwargs (dict): dictionary containing parameters for preprocessing
                                         (with_noise, noise_level, apply_logit, do_normalization,
                                          normalization)
        """

        # dataset specific
        self.layer_size = {'2': 9 * 16, '3': 18 * 50}[which_ds]
        self.depth = 45

        self.full_file = h5py.File(path_to_file, 'r')

        self.noise_level = preprocessing_kwargs.get('noise_level', 1e-4) # in MeV
        self.with_noise = preprocessing_kwargs.get('with_noise', True)

        showers = self.full_file['showers'][beginning_idx:beginning_idx+data_length]
        incident_energies = self.full_file['incident_energies']\
            [beginning_idx:beginning_idx+data_length]
        self.full_file.close()

        self.E_dep = showers.reshape(-1, self.depth, self.layer_size).sum(axis=-1)

        self.E_inc = incident_energies

    def __len__(self):
        # assuming file was written correctly
        return len(self.E_dep)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        energy_dep = self.E_dep[idx]
        e_inc = self.E_inc[idx]
        if self.with_noise:
            energy_dep = add_noise(energy_dep, noise_level=self.noise_level)

        sample = {'energy_dep': energy_dep, 'energy_inc':e_inc}

        return sample

class UpCaloData(Dataset):
    """ Dataloader for upcalo step"""
    def __init__(self, path_to_file, which_ds='2',
                 beginning_idx=0, data_length=100000,
                 **preprocessing_kwargs):
        """
        Args:
            path_to_file (string): path to .hdf5 file
            which_ds ('2' or '3'): which dataset (kind of redundant with path_to_file name)
            beginning_idx (int): at which index to start taking the data from
            data_length (int): how many events to take
            preprocessing_kwargs (dict): dictionary containing parameters for preprocessing
                                         (with_noise, noise_level, apply_logit, do_normalization,
                                          normalization)
        """

        # dataset specific
        self.num_radial = {'2': 9, '3': 18}[which_ds]
        self.num_alpha = {'2':16, '3': 50}[which_ds]
        self.depth = 45

        self.full_file = h5py.File(path_to_file, 'r')

        self.noise_level = preprocessing_kwargs.get('noise_level', 1e-4) # in MeV
        self.with_noise = preprocessing_kwargs.get('with_noise', True)

        showers = self.full_file['showers'][beginning_idx:beginning_idx+data_length]
        E_dep = showers.reshape(data_length, self.depth, 144).sum(axis=-1)
        E_dep = E_dep.reshape(data_length,int(self.depth/5),5)
        E_dep =  np.repeat(E_dep, int(self.num_alpha*self.num_radial/2),axis=1)
        self.E_dep = E_dep.reshape(data_length*int(self.depth*self.num_alpha*self.num_radial/10),5)


        showers = np.reshape(showers,(data_length,self.depth,self.num_alpha,self.num_radial))
        coarse_showers = np.reshape(showers,(data_length,int(self.depth/5),5,self.num_alpha,self.num_radial))
        coarse_showers = np.reshape(coarse_showers,(data_length,int(self.depth/5),5,int(self.num_alpha/2),2,9))
        coarse_showers = np.swapaxes(coarse_showers,2,3)
        coarse_showers = np.swapaxes(coarse_showers,4,5)
        coarse_showers = np.swapaxes(coarse_showers,3,4)
        coarse_showers = np.swapaxes(coarse_showers,2,3)

#################### conditioning on all neighbors ######################################

        coarse_adj_alpha_1 = np.roll(coarse_showers,1,axis=3)
        coarse_adj_alpha_2 = np.roll(coarse_showers,-1,axis=3)

        coarse_adj_r_1 = coarse_showers[:,:,:-1,:,:] #adjacent inner coarse voxel in r
        coarse_adj_r_1 = pad_front(coarse_adj_r_1,9,axis=2) #pad zeros for inner most coarse voxels

        coarse_adj_r_2 = coarse_showers[:,:,1:,:,:] #adjacent outer coarse voxel in r  
        coarse_adj_r_2 = pad_back(coarse_adj_r_2,9,axis=2) #pad zeros for outer most coarse voxels

        coarse_adj_z_1 = coarse_showers[:,:-1,:,:,:]
        coarse_adj_z_1 = pad_front(coarse_adj_z_1,9,axis=1)

        coarse_adj_z_2 = coarse_showers[:,1:,:,:,:]
        coarse_adj_z_2 = pad_back(coarse_adj_z_2,9,axis=1)
###########################################################################################
        
        incident_energies = self.full_file['incident_energies']\
            [beginning_idx:beginning_idx+data_length]
        self.full_file.close()
        self.fine_voxels = np.reshape(coarse_showers,(data_length*int(self.depth*self.num_alpha*self.num_radial/10),10))

#################### conditioning on all neighbors ######################################
        self.fine_alpha_1 = np.reshape(coarse_adj_alpha_1,(data_length*int(self.depth*self.num_alpha*self.num_radial/10),10)) 
        self.fine_alpha_2 = np.reshape(coarse_adj_alpha_2,(data_length*int(self.depth*self.num_alpha*self.num_radial/10),10))

        self.fine_r_1 = np.reshape(coarse_adj_r_1,(data_length*int(self.depth*self.num_alpha*self.num_radial/10),10))
        self.fine_r_2 =	np.reshape(coarse_adj_r_2,(data_length*int(self.depth*self.num_alpha*self.num_radial/10),10))

        self.fine_z_1 = np.reshape(coarse_adj_z_1,(data_length*int(self.depth*self.num_alpha*self.num_radial/10),10))
        self.fine_z_2 = np.reshape(coarse_adj_z_2,(data_length*int(self.depth*self.num_alpha*self.num_radial/10),10))
###########################################################################################
        
        coarse_z = []
        for i in range(9):
            coarse_z += list(i*np.ones(72))
        self.coarse_z = np.tile(coarse_z,data_length)

        r_label = []
        for i in range(9):
            for j in range(9):
                r_label += list(j*np.ones(8))
        self.r_label = np.tile(r_label,data_length)

        self.E_inc = np.repeat(incident_energies, int(self.depth*self.num_alpha*self.num_radial/10))

    def __len__(self):
        # assuming file was written correctly
        return len(self.coarse_z)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        e_inc = self.E_inc[idx]
        fine_voxels = self.fine_voxels[idx]
        energy_dep = self.E_dep[idx]

        #################### conditioning on all neighbors ######################################
        fine_alpha_1 = self.fine_alpha_1[idx]
        fine_alpha_2 = self.fine_alpha_2[idx]
        fine_r_1 = self.fine_r_1[idx]
        fine_r_2 = self.fine_r_2[idx]
        fine_z_1 = self.fine_z_1[idx]
        fine_z_2 = self.fine_z_2[idx]
        ###########################################################################################

        z_labels  = self.coarse_z[idx]
        r_labels = self.r_label[idx]

        if self.with_noise:
            fine_voxels = add_noise(fine_voxels, noise_level=self.noise_level)
        coarse_voxels = fine_voxels.sum(axis=-1)

        #################### conditioning on all neighbors ######################################
        coarse_energy_adj_alpha_1 = fine_alpha_1.sum(axis=-1)
        coarse_energy_adj_alpha_2 = fine_alpha_2.sum(axis=-1)
        coarse_energy_adj_r_1 = fine_r_1.sum(axis=-1)
        coarse_energy_adj_r_2 = fine_r_2.sum(axis=-1)
        coarse_energy_adj_z_1 = fine_z_1.sum(axis=-1)
        coarse_energy_adj_z_2 = fine_z_2.sum(axis=-1)
        ###########################################################################################
        sample = {'z_labels': z_labels, 'r_labels': r_labels, 'coarse_voxels': coarse_voxels,'E_adj_r_1': coarse_energy_adj_r_1,'E_adj_r_2': coarse_energy_adj_r_2,'E_adj_z_1': coarse_energy_adj_z_1,'E_adj_z_2': coarse_energy_adj_z_2, 'E_adj_alpha_1': coarse_energy_adj_alpha_1, 'E_adj_alpha_2': coarse_energy_adj_alpha_2, 'fine_voxels': fine_voxels, 'energy_dep': energy_dep, 'energy_inc':e_inc}

        return sample


class CaloDataShowerShape(Dataset):
    """ Dataloader for every Calorimeter Layer (flow-2)"""
    def __init__(self, path_to_file, which_ds='2',
                 beginning_idx=0, data_length=100000,
                 **preprocessing_kwargs):
        """
        Args:
            path_to_file (string): path to .hdf5 file
            which_ds ('2' or '3'): which dataset (kind of redundant with path_to_file name)
            beginning_idx (int): at which index to start taking the data from
            data_length (int): how many events to take
            preprocessing_kwargs (dict): dictionary containing parameters for preprocessing
                                         (with_noise, noise_level, apply_logit, do_normalization,
                                          normalization)
        """

        # dataset specific
        self.layer_size = {'2': 9 * 16, '3': 18 * 50}[which_ds]
        # dataset specific
        self.num_radial = {'2': 9, '3': 18}[which_ds]
        self.num_alpha = {'2':16, '3': 50}[which_ds]
        self.depth = 45

        self.full_file = h5py.File(path_to_file, 'r')

        self.noise_level = preprocessing_kwargs.get('noise_level', 1e-4) # in MeV
        self.apply_logit = preprocessing_kwargs.get('apply_logit', True)
        self.with_noise = preprocessing_kwargs.get('with_noise', True)
        self.do_normalization = preprocessing_kwargs.get('do_normalization', True)

        showers = self.full_file['showers'][beginning_idx:beginning_idx+data_length]
        incident_energies = self.full_file['incident_energies']\
            [beginning_idx:beginning_idx+data_length]
        self.full_file.close()

        self.E_inc = incident_energies

        showers = np.reshape(showers,(data_length,self.depth,self.num_alpha,self.num_radial))

        showers = showers.reshape(data_length,9,5,self.num_alpha,self.num_radial)
        showers = showers.reshape(data_length,9,5,int(self.num_alpha/2),2,self.num_radial)
        showers = np.swapaxes(showers,2,3)
        showers = np.swapaxes(showers,4,5)
        showers = np.swapaxes(showers,3,4)
        showers = np.swapaxes(showers,2,3)
        showers = showers.reshape(data_length,9,int(self.num_alpha*self.num_radial/2),10)
        showers = showers.sum(axis=-1)
        showers = add_noise(showers, noise_level=self.noise_level)

        showers_E = showers.sum(axis=-1)
        showers_norm = showers/28048
        showers_norm = showers_norm.reshape(data_length, 648)

        self.Edep = showers_E
        self.coarse_voxels_norm = showers_norm

    def __len__(self):
        """ length of dataset should be length of E_inc """
        return len(self.E_inc)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        energy = self.E_inc[idx]
        energy_dep = self.Edep[idx]
        coarse_voxels = self.coarse_voxels_norm[idx]

        sample = {'energy': energy,'energy_dep': energy_dep, 'coarse_voxels': coarse_voxels}

        return sample

def get_calo_dataloader(path_to_file, which_flow, device, which_ds='2', batch_size=32,
                        small_file=False, **preprocessing_kwargs):
    """ returns train/test dataloader for training each of the flows """
    kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == 'cuda' else {}

    if small_file:
        data_length = 50000
        train_length = 40000
        test_length = 10000
    else:
        data_length = 100000
        train_length = int(0.7*data_length)
        test_length = int(0.3*data_length)

    if which_flow == 1:
        train_dataset = CaloDataLayerEnergy(path_to_file, which_ds=which_ds,
                                            beginning_idx=0, data_length=train_length,
                                            **preprocessing_kwargs)
        test_dataset = CaloDataLayerEnergy(path_to_file, which_ds=which_ds,
                                           beginning_idx=train_length, data_length=test_length,
                                           **preprocessing_kwargs)
    elif which_flow == 2:
        train_dataset = CaloDataShowerShape(path_to_file, which_ds=which_ds,
                                            beginning_idx=0, data_length=train_length,
                                            **preprocessing_kwargs)
        test_dataset = CaloDataShowerShape(path_to_file, which_ds=which_ds,
                                           beginning_idx=train_length, data_length=test_length,
                                           **preprocessing_kwargs)
    elif which_flow == 3:
        train_dataset = UpCaloData(path_to_file, which_ds=which_ds,
                                            beginning_idx=0, data_length=train_length,
                                            **preprocessing_kwargs)
        test_dataset = UpCaloData(path_to_file, which_ds=which_ds,
                                           beginning_idx=train_length, data_length=test_length,
                                           **preprocessing_kwargs)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, **kwargs)
    return train_dataloader, test_dataloader

def get_coarse_voxels(path_to_file, device, which_ds='2', batch_size=32,
                        small_file=False, **preprocessing_kwargs):
    """ returns coarse voxel information (+conditionals) required for upsampling """
    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}

    if small_file:
        data_length = 50000

    else:
        data_length = 100000

    coarse_showers = UpCaloData(path_to_file, which_ds=which_ds,
                                            beginning_idx=0, data_length=data_length,
                                            **preprocessing_kwargs)
    coarse_data = DataLoader(coarse_showers, batch_size=batch_size,
                                  shuffle=False, **kwargs)
    return coarse_data

def get_flow2_data(path_to_file, device, which_ds='2', batch_size=32,
                        small_file=False, **preprocessing_kwargs):
    """ returns true conditional info required to generate from flow-II """
    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}

    if small_file:
        data_length = 50000

    else:
        data_length = 100000

    showers = CaloDataShowerShape(path_to_file, which_ds=which_ds,
                                           beginning_idx=0, data_length=data_length,
                                           **preprocessing_kwargs)
    data = DataLoader(showers, batch_size=batch_size,
                                  shuffle=False, **kwargs)
    return data
