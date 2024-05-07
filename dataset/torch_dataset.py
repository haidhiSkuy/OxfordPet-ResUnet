import cv2  
import torch 
import numpy as np
from torch.utils.data import Dataset 

 

def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask

class OxfordPetDataset(Dataset): 
    def __init__(self, image_list, mask_list, transform) :
        self.image_list = image_list 
        self.mask_list = mask_list 
        self.transform = transform 
    
    def __len__(self): 
        return len(self.image_list) 
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        mask = cv2.imread(self.mask_list[index], cv2.IMREAD_UNCHANGED) 
        mask = preprocess_mask(mask) 

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask
