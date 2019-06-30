import os
import random
from random import shuffle
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from utils import rle2mask
from skimage.transform import resize
import cv2
from pathlib import Path

# encode = list(encode_df.loc[encode_df['ImageId'] == '.'.join(f.split('/')[-1].split('.')[:-1]),
#                                ' EncodedPixels'].values)
            
# encode = get_mask(encode,img.shape[1],img.shape[0])
# encode = resize(encode,(img_size,img_size))

class SIIMDataset(data.Dataset):

	def __init__(self,rle_csv,data_dir, image_size = 224, mode = 'train', augmentation_prob =0.4):
		self.rle_csv = rle_csv
		self.data_dir = data_dir
		self.dataframe = pd.read_csv(rle_csv)
		
		# self.dataframe = dataframe.drop( dataframe[ not os.path.isfile(os.path.join(self.data_dir,dataframe.ImageId.tolist()  \
		# 																				+('.png')) ].index )
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		self.ImageIds = self.dataframe.iloc[:,0].values
		self.imgs = os.listdir(self.data_dir)

	def get_mask(self, encode, width, height):
		if encode ==[] or encode == ' -1':
			return rle2mask(' -1',width,height)
		else :
			return rle2mask(encode[0],width,height)       			# Isnt it just returning one of the mask not all.

	def __len__(self):
		return len(os.listdir(self.data_dir))

	def __getitem__(self, index):
		
	    image_path = self.imgs[index]
	    image_path = os.path.join(self.data_dir, image_path)
	    # print("Selected Imagee : ",image_path)
	    # print(type(image_path))

	    image = cv2.imread(image_path) 
	    # print(image.shape)
	    # if(image.shape[2]<3):
	    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    image = cv2.resize(image, (self.image_size,self.image_size))
	    # print(type(image))
	   
		
	    encode = list(self.dataframe.loc[self.dataframe['ImageId'] == '.'.join(image_path.split('/')[-1].split('.')[:-1]),
                               ' EncodedPixels'].values)

	    encode = self.get_mask(encode,image.shape[1],image.shape[0])
	    # encode = np.expand_dims(encode, axis=2)
	    print(encode.shape)
	    encode = resize(encode,(self.image_size,self.image_size))				# 1 Here means the channel dimension of 1.
	    # print(type(encode))
	    # 

	    image = Image.fromarray(image.astype('uint32'), 'RGB')
	    encode = Image.fromarray(encode.astype('uint32'), '1')
	    
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
		    encode = Transform(encode)

		    ShiftRange_left = random.randint(0,20)
		    ShiftRange_upper = random.randint(0,20)
		    ShiftRange_right = image.size[0] - random.randint(0,20)
		    ShiftRange_lower = image.size[1] - random.randint(0,20)
		    image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
		    encode = encode.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

		    if random.random() < 0.5 :
			    image = F.hflip(image)
			    encode = F.hflip(encode)

		    if random.random() < 0.5:
			    image = F.vflip(image)
			    encode = F.vflip(encode)

		    Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

		    image = Transform(image)

		    Transform =[]


	    Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
	    Transform.append(T.ToTensor())
	    Transform = T.Compose(Transform)
		
	    image = Transform(image)
	    encode = Transform(encode)

	    Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	    image = Norm_(image)
		
	    print(image.shape)
	    print(encode.shape)
	    return image, encode


def get_loader(rle_csv,data_dir,batch_size ,image_size = 224, mode = 'train', augmentation_prob =0.4, num_workers=2):
	"""Builds and returns Dataloader."""
	
	dataset = SIIMDataset(data_dir = data_dir, rle_csv = rle_csv, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader


    
	