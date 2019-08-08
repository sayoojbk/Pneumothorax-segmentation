# %% [code]
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd 
import os
import collections
from tqdm import tqdm
import random
import torch.utils.data
from torchvision import transforms
import torchvision
from PIL import Image
from unet1_utils import rle2mask



class SiimDataset(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, df_path, img_dir, transform=None):
        self.df = pd.read_csv(df_path)
        self.height = 1024
        self.width = 1024
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)
        self.transform = transform
        
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
        # mask = torch.as_tensor(mask, dtype=torch.uint8)

        if self.transform:
            # Convert PIL image to numpy array
            image_np = np.array(img)
            # Apply transformations
            augmented = self.transform(image=image_np , mask=mask) 
            # Returns the numpy array of mask and images.
            image = augmented['image']
            mask  = augmented['mask']

        mask = mask/255
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        return image, mask


    def __len__(self):
        return len(self.image_info)


from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,Resize,Rotate , VerticalFlip,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop,
    ChannelShuffle,RandomRotate90
)


albumentations_transform = Compose([
    Resize(256,256),
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),

    Rotate(limit=45, p=0.5),
    VerticalFlip(p=0.5),
    ChannelShuffle(p=0.3),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    
    RandomSizedCrop(min_max_height=(128, 256), height=256, width=256,p=0.5),
    ToFloat(max_value=1)
])

def create_dataset():
    dataset_train = SiimDataset(
                        df_path="../input/siim-dicom-images/train-rle.csv", 
                        img_dir = "../input/siim-png-images/input/train_png/", 
                        transform=albumentations_transform)

    return dataset_train