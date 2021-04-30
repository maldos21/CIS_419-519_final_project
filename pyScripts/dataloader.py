import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from skimage import io 

import os
import pandas as pd
from torchvision.io import read_image

class CircuitBoardImageDataset(Dataset) :
    def __init__(self, annotations_file, img_dir, transform=None) :
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.annotation_file = annotations_file
        
    def __len__(self) :
        return len(self.img_labels)
    
    def __getitem__(self, idx) :
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        label = self.img_labels.iloc[idx, 1]
        pcb = self.img_labels.iloc[idx, 2]
        if self.transform :
            image = self.transform(image)
        sample = (image, label)
        return sample
    
    def getItemInfo(self, idx) :
        img_path = self.img_labels.iloc[idx, 0].split("\\")[1]
        pcb = self.img_labels.iloc[idx, 2]
        return pcb, img_path