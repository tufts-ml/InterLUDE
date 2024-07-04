import numpy as np
import os
from PIL import Image

# from torchvision import transforms

# from .randaugment import RandAugmentMC



class STL10:
    def __init__(self, dataset_path, transform_fn=None):
        self.dataset = np.load(dataset_path, allow_pickle=True).item() #need to use HWC version of the data
        self.transform_fn = transform_fn
        
        #the STL dataset is (3, 96, 96) when downloaded, need to transpose to (96, 96, 3); CIFAR10/100 is already (32, 32, 3) so no need to transpose
        self.dataset_images = self.dataset["images"].transpose(0,2,3,1)
        self.dataset_labels = self.dataset["labels"]
        
        
    def __getitem__(self, idx):
#         image = self.dataset["images"][idx]
#         label = self.dataset["labels"][idx]
        image = self.dataset_images[idx]
        label = self.dataset_labels[idx]
        
        image = Image.fromarray(image)
        if self.transform_fn is not None:
            image = self.transform_fn(image)
            
        return image, label

    def __len__(self):
        return len(self.dataset_images)
    
