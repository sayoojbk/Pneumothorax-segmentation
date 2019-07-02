from __future__ import print_function

from collections import defaultdict, deque
import datetime
import pickle
import time
import torch.distributed as dist
import errno
from torch.utils import data
import collections
import os
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from PIL import Image, ImageFile
import pandas as pd
from torchvision import transforms
import torchvision
import random

from utils import rle2mask

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SIIMDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_dir):
        self.df = pd.read_csv(df_path)
        self.height = 1024
        self.width = 1024
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)

        counter = 0
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, image_id)
            if os.path.exists(image_path + '.png') and row[" EncodedPixels"].strip() != "-1":
                self.image_info[counter]["image_id"] = image_id
                self.image_info[counter]["image_path"] = image_path
                self.image_info[counter]["annotations"] = row[" EncodedPixels"].strip()
                counter += 1

    def __getitem__(self, idx):
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path + '.png').convert("RGB")
        width, height = img.size
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        info = self.image_info[idx]

        mask = rle2mask(info['annotations'], width, height)
        mask = Image.fromarray(mask.T)
        mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
        mask = np.expand_dims(mask, axis=0)

        mask = torch.as_tensor(mask, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        
        img = transforms.ToTensor()(img)
        
        if random.random() < 0.5:
            height, width = img.shape[-2:]
            img = img.flip(-1)
            mask = mask.flip(-1)
        
        return img, mask

    def __len__(self):
        return len(self.image_info)



def get_loader(data_dir, rle_csv,batch_size, num_workers=2):
	"""Builds and returns Dataloader."""
	
	dataset = SIIMDataset(img_dir = data_dir, df_path = rle_csv)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader


    
	
