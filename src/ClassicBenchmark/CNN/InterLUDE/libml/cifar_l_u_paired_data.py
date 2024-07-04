import numpy as np
import os
from PIL import Image
import random
import torch

# from torchvision import transforms

# from .randaugment import RandAugmentMC



class CIFAR_l_u_paired:
    def __init__(self, l_dataset_path, u_dataset_path, u_batchsize_multiplier, transform_fn=None):
        
        self.l_dataset = np.load(l_dataset_path, allow_pickle=True).item() #need to use HWC version of the data
        self.u_dataset = np.load(u_dataset_path, allow_pickle=True).item() #need to use HWC version of the data
        
        self.u_batchsize_multiplier = u_batchsize_multiplier
        
        self.l_size = len(self.l_dataset["images"])
        self.u_size = len(self.u_dataset["images"])
        
        self.u_indices = list(range(self.u_size))
        
        self.transform_fn = transform_fn
        
    def __getitem__(self, idx):
        
        assert self.transform_fn is not None
        
        l_image = self.l_dataset["images"][idx]
        l_label = self.l_dataset["labels"][idx]
        
        #get u_batchsize_multiplier unlabeled sample for this 1 labeled sample
        u_idx = random.sample(self.u_indices, self.u_batchsize_multiplier)
        
#         print('this call, l_idx: {} u_idx: {}'.format(idx, u_idx))
        
        
        l_image = Image.fromarray(l_image)
        l_weak, l_strong = self.transform_fn(l_image)
        
        
        u_weaks, u_strongs = [], []
        for i in u_idx:
            u_image = self.u_dataset["images"][i]
            u_image = Image.fromarray(u_image)
            u_weak, u_strong = self.transform_fn(u_image)
            u_weaks.append(u_weak)
            u_strongs.append(u_strong)
        
        # Stack along a new dimension to get [7, C, H, W] correspond to each labeled samples
        u_weaks = torch.stack(u_weaks, dim=0)
        u_strongs = torch.stack(u_strongs, dim=0)


        return (l_weak, l_strong, l_label), (u_weaks, u_strongs)
    
    
    
    def __len__(self):
        return self.l_size #Without explicitly providing a sampler, the DataLoader determines the range of indices based on the length of the dataset. Every PyTorch Dataset should implement the __len__ method, which returns the number of items in the dataset. The DataLoader uses this method to determine the range of valid indices for the dataset.


        
    
    
