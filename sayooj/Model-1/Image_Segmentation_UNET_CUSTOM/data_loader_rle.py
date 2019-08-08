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
from torchvision import transforms as T
from torchvision.transforms import functional as F
from random import shuffle

from utils import rle2mask

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SIIMDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_dir,mode, augmentation_prob):
        self.df = pd.read_csv(df_path)
        self.height = 224
        self.width = 224
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob

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
        image = img.resize((self.width, self.height), resample=Image.BILINEAR)
        info = self.image_info[idx]


        mask = rle2mask(info['annotations'], width, height)
        mask = Image.fromarray(mask.T)
        mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
        mask = np.expand_dims(mask, axis=0)

        
        image_id = torch.tensor([idx])

        aspect_ratio = image.size[1]/image.size[0]

		
        Transform = []
        ResizeRange = random.randint(300,320)

        Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
        p_transform = random.random()


        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
	
            RotationDegree = random.randint(0,3)
            RotationDegree = self.RotationDegree[RotationDegree]
		    
            
            if (RotationDegree == 90) or (RotationDegree == 270):
			    
                aspect_ratio = 1/aspect_ratio


            Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
            RotationRange = random.randint(-10,10)
            Transform.append(T.RandomRotation((RotationRange,RotationRange)))
            CropRange = random.randint(250,270)
            Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
            Transform = T.Compose(Transform)
			
            image = Transform(image)
            mask = Transform(mask)

            ShiftRange_left = random.randint(0,20)
            ShiftRange_upper = random.randint(0,20)
            ShiftRange_right = image.size[0] - random.randint(0,20)
            ShiftRange_lower = image.size[1] - random.randint(0,20)
            image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
            mask = mask.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

			
            if random.random() < 0.5:
			
            	image = F.hflip(image)
			
            	mask = F.hflip(mask)

			
            if random.random() < 0.5:
				
                image = F.vflip(image)
				
                mask = F.vflip(mask)

            Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)
            image = Transform(image)
            Transform =[]


        Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
		
		
        image = Transform(image)
		
        mask = Transform(mask)
        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		
        image = Norm_(image)
        return image, mask


    def __len__(self):
        return len(self.image_info)



def get_loader(data_dir, rle_csv,batch_size, num_workers=2,mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = SIIMDataset(img_dir = data_dir, df_path = rle_csv,mode='train',augmentation_prob=0.4)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader


    
	
