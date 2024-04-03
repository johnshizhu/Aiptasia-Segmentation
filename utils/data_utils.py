import torch
from torch.utils.data import Dataset
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
        folder_path = self.root_dir + "/"+ str(idx+1)

        jpg_file_paths = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

        image_path = ""
        label_path = ""

        for i in jpg_file_paths:
            if i[0] == "I":
                image_path = folder_path + '\\' + i
            if i[0] == "l":
                label_path = folder_path + '\\' + i

        image = cv2.imread(image_path)
        # Load label as grayscale
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image = image / 255
        label[label < 50] = 0
        label[label > 200] = 1

        torch_image = torch.from_numpy(image)
        torch_image = torch_image.permute(2,0,1)

        torch_label = torch.from_numpy(label)
        torch_label = torch_label.unsqueeze(0)

        return torch_image.float(), torch_label.float()