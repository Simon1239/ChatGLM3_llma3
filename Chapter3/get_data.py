#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : get_data.py

import csv
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from modelscope.msdatasets import MsDataset

class DatasetLoader(Dataset):
    def __init__(self, data):
        self.data = data

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return image_transform(image)
    
    def __getitem__(self, index):
        image_path, label = self.data[index]['image:FILE'], self.data[index]['category']
        image = self.preprocess_image(image_path)
        label = self.data[index][1]
        return image, int(label)
    
    def __len__(self):
        return len(self.data)

ms_train_dataset = MsDataset.load(
    'cats_and_dogs', namespace='tany0699',
    subset_name='default', split='train'
)
data_train = DatasetLoader(ms_train_dataset)