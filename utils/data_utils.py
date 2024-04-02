import torch
from torch.utils.data import Dataset
import os
import cv2
import glob
import numpy as np

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
        folder_path = self.root_dir + str(idx)

        jpg_file_paths = glob.glob(f'{folder_path}/*.jpg')

        image_path = ""
        label_path = ""

        for i in jpg_file_paths:
            if i[0] == "I":
                image_path = i
            if i[0] == "L":
                label_path = i

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        print(f'image shape: {image.shape}')
        print(f'label shape: {label.shape}')

        print(np.min(image))
        print(np.min(label))

        print(np.max(image))
        print(np.max(label))

        # Normalize Image and Label inforamtion
        image = image / 255
        label = label / 255

        return image, label