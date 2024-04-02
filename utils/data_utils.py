import torch
from torch.utils.data import Dataset
import os
import cv2

class AiptasiaDataset(Dataset):
    def __init__(self, root_dir):
        '''
        Pytorch Dataset for Aiptasia Microscopic image data
        root_dir --> Root Directory of Dataset
        '''
        self.root_dir = root_dir
        self.samples = [os.path.join(root_dir, o) for o in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, o))]
 
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, f'image_{idx}.jpg')
        label_path = os.path.join(self.root_dir, f'label_{idx}.jpg')

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # Normalize Image and Label inforamtion
        image = image / 255
        label = label / 255

        return image, label