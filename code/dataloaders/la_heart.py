import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from PIL import ImageFilter
import random
from torchvision import transforms
import cv2
from functools import wraps

class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train',train_flod=None, common_transform=None,sp_transform=None):
        self._base_dir = base_dir
        self.common_transform = common_transform
        self.sp_transform = sp_transform
        self.sample_list = []
        # print(train_flod)
        if split=='train':
            with open(self._base_dir+'/'+train_flod, 'r') as f:
                self.image_list = f.readlines()
        elif split=='eval':
            with open(self._base_dir+'/'+train_flod, 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]

        print("total {} unlabel_samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/data/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.common_transform:
            sample = self.common_transform(sample)
        if self.sp_transform: 
            sample1 = self.sp_transform(sample)
            # sample2 = self.sp_transform(sample)
            # return [sample1,] # sample2
            return sample1
        else:
            return  sample
        
class PCT(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train',train_flod=None, common_transform=None,sp_transform=None):
        self._base_dir = base_dir
        self.common_transform = common_transform
        self.sp_transform = sp_transform
        self.sample_list = []
        # print(train_flod)
        if split=='train':
            with open(self._base_dir+'/'+train_flod, 'r') as f:
                self.image_list = f.readlines()
            # print(len(self.image_list))
            # print(self.image_list[0])
            # print(self.image_list)
        elif split=='eval':
            with open(self._base_dir+'/'+train_flod, 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]

        print("total {} unlabel_samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/data/"+image_name+"_norm.h5", 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.common_transform:
            sample = self.common_transform(sample)
        if self.sp_transform: 
            sample1 = self.sp_transform(sample)
            # sample2 = self.sp_transform(sample)
            # return [sample1,] # sample2
            return sample1
        else:
            return  sample
        
class BRATS(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train',train_flod=None, common_transform=None,sp_transform=None):
        self._base_dir = base_dir
        self.common_transform = common_transform
        self.sp_transform = sp_transform
        self.sample_list = []
        # print(train_flod)
        if split=='train':
            with open(self._base_dir+'/'+train_flod, 'r') as f:
                self.image_list = f.readlines()
            # print(len(self.image_list))
            # print(self.image_list[0])
            # print(self.image_list)
        elif split=='eval':
            with open(self._base_dir+'/'+train_flod, 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]

        print("total {} unlabel_samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # print(idx)
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/data/"+image_name+".h5", 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.common_transform:
            sample = self.common_transform(sample)
        if self.sp_transform: 
            sample1 = self.sp_transform(sample)
            # sample2 = self.sp_transform(sample)
            # return [sample1,] # sample2
            return sample1
        else:
            return  sample

class WeakStrongAugment(object):
    def __init__(self, p_color=0.8, p_blur=0.2, flag_rot=True):
        # self.output_size = output_size
        self.randrotflip = RandomRotFlip()
        # self.randcrop = RandomCrop(output_size)
        self.randcolor = RandomBrightnessContrast(brightness_limit=0.5, 
                                                  contrast_limit=0.5, 
                                                  prob=p_color)
        self.randblur = RandomGaussianNoise(sigma=[0.1, 1.0], 
                                            apply_prob=p_blur)
        # self.totensor = ToTensor()
        self.flag_rot = flag_rot

    def __call__(self, sample):
        # rand rot flip
        # if self.flag_rot:
        sample = self.randrotflip(sample)
        # rand crop
        # sample = self.randcrop(sample)

        # get image, labels
        image_weak, label = sample["image"], sample["label"]
        image_strong = image_weak.copy()
        # apply color aug
        image_strong = self.randcolor(image_strong)
        # apply blur
        image_strong = self.randblur(image_strong)

        # to tensor
        # image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        # image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.uint8)).long()

        new_sample = {
            "image_weak": image_weak,
            "image_strong": image_strong,
            # "label_aug": label,
            "label": label,
        }
        return new_sample

class Normalise(object):
    def __call__(self, sample):
        image = sample['image']
        return{'image': (image - image.min()) / (image.max()-image.min()), 'label': sample['label']}

class WeakStrongAugment_PCT(object):
    def __init__(self, p_color=0.8, p_blur=0.2, flag_rot=True):
        # self.output_size = output_size
        # self.randrotflip = RandomRotFlip()
        # self.randcrop = RandomCrop(output_size)
        self.randcolor = RandomBrightnessContrast(brightness_limit=0.5, 
                                                  contrast_limit=0.5, 
                                                  prob=p_color)
        self.randblur = RandomGaussianNoise(sigma=[0.1, 1.0], 
                                            apply_prob=p_blur)
        # self.totensor = ToTensor()
        self.flag_rot = flag_rot

    def __call__(self, sample):
        # rand rot flip
        # if self.flag_rot:
        # sample = self.randrotflip(sample)
        # rand crop
        # sample = self.randcrop(sample)

        # get image, labels
        image_weak, label = sample["image"], sample["label"]
        image_strong = image_weak.copy()
        # apply color aug
        image_strong = self.randcolor(image_strong)
        # apply blur
        image_strong = self.randblur(image_strong)

        # image_strong = (image_strong - image_strong.min()) / (image_strong.max()-image_strong.min())

        # to tensor
        # image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        # image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.uint8)).long()

        new_sample = {
            "image_weak": image_weak,
            "image_strong": image_strong,
            # "label_aug": label,
            "label": label,
        }
        return new_sample
    
class WeakStrongAugment_BRATS(object):
    def __init__(self, p_color=0.8, p_blur=0.2, flag_rot=True):
        # self.output_size = output_size
        self.randrotflip = RandomRotFlip()
        # self.randcrop = RandomCrop(output_size)
        self.randcolor = RandomBrightnessContrast(brightness_limit=0.5, 
                                                  contrast_limit=0.5, 
                                                  prob=p_color)
        self.randblur = RandomGaussianNoise(sigma=[0.1, 1.0], 
                                            apply_prob=p_blur)
        # self.totensor = ToTensor()
        self.flag_rot = flag_rot

    def __call__(self, sample):
        # rand rot flip
        # if self.flag_rot:
        sample = self.randrotflip(sample)
        # rand crop
        # sample = self.randcrop(sample)

        # get image, labels
        image_weak, label = sample["image"], sample["label"]
        image_strong = image_weak.copy()
        # apply color aug
        image_strong = self.randcolor(image_strong)
        # apply blur
        image_strong = self.randblur(image_strong)

        # image_strong = (image_strong - image_strong.min()) / (image_strong.max()-image_strong.min())

        # to tensor
        # image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        # image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.uint8)).long()

        new_sample = {
            "image_weak": image_weak,
            "image_strong": image_strong,
            # "label_aug": label,
            "label": label,
        }
        return new_sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

class PancreasDataset(Dataset):
    """ LA Dataset
        input: base_dir -> your parent level path
               split -> "sup", "unsup" and "eval", must specified
    """
    def __init__(self, base_dir, data_dir,
                 split, num=None, config=None):
        self.data_dir = data_dir
        self._base_dir = base_dir
        self.sample_list = []
        if split == 'test':
            with open(self._base_dir+'/test.list', 'r') as f:
                self.image_list = f.readlines()
        if split == 'train':
            with open(self._base_dir+'/train.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.strip() for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        self.aug = True if split != 'test' else False
        self.training_transform = transforms.Compose([
                                    Normalise(),
                                    RandomCrop((96, 96, 96)),
                                    ToTensor(),
                                ])
        self.testing_transform = transforms.Compose([
            # Normalise()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self.data_dir+"/data/"+image_name+"_norm.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if not self.aug:
            sample = self.testing_transform(sample)
            return sample['image'], sample['label']
        return self.training_transform(sample)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        cons_image = sample['cons_image']
        cons_image = cons_image.reshape(1, cons_image.shape[0], cons_image.shape[1], cons_image.shape[2]).astype(np.float32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                'x_range': sample['normal_range_x'], 'y_range': sample['normal_range_y'],
                'z_range': sample['normal_range_z']}, \
               {'image': torch.from_numpy(cons_image), 'label': torch.from_numpy(sample['cons_label']).long(),
                'x_range': sample['cons_range_x'], 'y_range': sample['cons_range_y'], 'z_range': sample['cons_range_z']}

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def calculate_subvolume_size(self, w, h, d, volume_ratio=0.8):
    # Calculate the target volume of the subvolume
        target_volume = w * h * d * volume_ratio
        
        # Randomly generate the dimensions of the subvolume
        # Ensure that the product of the dimensions is close to the target volume
        while True:
            sub_w = np.random.randint(int(w * 0.6), w)  # Random width, at least 50% of original
            sub_h = np.random.randint(int(h * 0.6), h)  # Random height, at least 50% of original
            sub_d = np.random.randint(int(d * 0.9), d)  # Random depth, at least 50% of original
            if np.abs(sub_w * sub_h * sub_d - target_volume) / target_volume < 0.1:  # Allow 10% tolerance
                break
        
        return sub_w, sub_h, sub_d

    # 本来想使用resize这种技巧的，但是发现如果使用resize就与test的方法不能保持一致了，还是得相信前辈的代码
    def generate_subvolumes(self, w, h, d, sub_w=112, sub_h=112, sub_d=80):
        # Ensure the subvolume size is not larger than the original volume
        if sub_w > w or sub_h > h or sub_d > d:
            raise ValueError("Subvolume size cannot be larger than the original volume.")
        
        # while True:
        # Randomly generate the starting coordinates for the first subvolume
        start_x1 = np.random.randint(0, w - sub_w + 1)
        start_y1 = np.random.randint(0, h - sub_h + 1)
        start_z1 = np.random.randint(0, d - sub_d + 1)
        
        # Randomly generate the starting coordinates for the second subvolume
        # Ensure that there is an overlap with the first subvolume
        start_x2 = np.random.randint(max(0, start_x1 - sub_w + 1), min(w - sub_w, start_x1 + sub_w - 1) + 1)
        start_y2 = np.random.randint(max(0, start_y1 - sub_h + 1), min(h - sub_h, start_y1 + sub_h - 1) + 1)
        start_z2 = np.random.randint(max(0, start_z1 - sub_d + 1), min(d - sub_d, start_z1 + sub_d - 1) + 1)
        
        # Calculate the overlapping region
        overlap_x_start = max(start_x1, start_x2)
        overlap_y_start = max(start_y1, start_y2)
        overlap_z_start = max(start_z1, start_z2)
        
        overlap_x_end = min(start_x1 + sub_w, start_x2 + sub_w)
        overlap_y_end = min(start_y1 + sub_h, start_y2 + sub_h)
        overlap_z_end = min(start_z1 + sub_d, start_z2 + sub_d)

        sub_wl = overlap_x_end - overlap_x_start
        sub_hl = overlap_y_end - overlap_y_start
        sub_dl = overlap_z_end - overlap_z_start

        # target_volume = sub_w * sub_h * sub_d * 0.8
        # sub_wflag = self.output_size[0] * 0.7 <= sub_wflag <= self.output_size[0]
        # sub_hflag = self.output_size[1] * 0.7 <= sub_hflag <= self.output_size[1]
        # sub_dflag = self.output_size[2] * 0.7 <= sub_dflag <= self.output_size[2]
        # if (np.abs(sub_wl * sub_hl * sub_dl - target_volume) / target_volume < 0.1) and sub_wflag and sub_hflag and sub_dflag: 
        #     break
        
        # Calculate the coordinates of the overlapping region in the subvolumes
        overlap_in_sub1_x_start = overlap_x_start - start_x1
        overlap_in_sub1_y_start = overlap_y_start - start_y1
        overlap_in_sub1_z_start = overlap_z_start - start_z1
        
        overlap_in_sub2_x_start = overlap_x_start - start_x2
        overlap_in_sub2_y_start = overlap_y_start - start_y2
        overlap_in_sub2_z_start = overlap_z_start - start_z2
        
        # Return the coordinates of the subvolumes and the overlapping region
        sub1_coords = (start_x1, start_y1, start_z1, sub_w, sub_h, sub_d)
        sub2_coords = (start_x2, start_y2, start_z2, sub_w, sub_h, sub_d)
        
        overlap_coords_in_sub1 = (overlap_in_sub1_x_start, overlap_in_sub1_y_start, overlap_in_sub1_z_start, 
                                overlap_x_end - overlap_x_start, overlap_y_end - overlap_y_start, overlap_z_end - overlap_z_start)
        
        overlap_coords_in_sub2 = (overlap_in_sub2_x_start, overlap_in_sub2_y_start, overlap_in_sub2_z_start, 
                                overlap_x_end - overlap_x_start, overlap_y_end - overlap_y_start, overlap_z_end - overlap_z_start)
        
        return sub1_coords, sub2_coords, overlap_coords_in_sub1, overlap_coords_in_sub2

    def __call__(self, sample):
        image_weak, image_strong, label = sample['image_weak'], sample['image_strong'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image_weak = np.pad(image_weak, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image_strong = np.pad(image_strong, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image_weak.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        # w1 = np.random.randint(0, w - self.output_size[0])
        # h1 = np.random.randint(0, h - self.output_size[1])
        # d1 = np.random.randint(0, d - self.output_size[2])
        # label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        # image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        sub1_coords, sub2_coords, overlap_coords_in_sub1, overlap_coords_in_sub2 = self.generate_subvolumes(w, h, d,  sub_w=self.output_size[0], sub_h=self.output_size[1], sub_d=self.output_size[2])
        cons_image_weak = image_weak[sub1_coords[0]:sub1_coords[0] + sub1_coords[3], sub1_coords[1]:sub1_coords[1] + sub1_coords[4], sub1_coords[2]:sub1_coords[2] + sub1_coords[5]]
        cons_image_strong = image_strong[sub1_coords[0]:sub1_coords[0] + sub1_coords[3], sub1_coords[1]:sub1_coords[1] + sub1_coords[4], sub1_coords[2]:sub1_coords[2] + sub1_coords[5]]
        cons_label = label[sub1_coords[0]:sub1_coords[0] + sub1_coords[3], sub1_coords[1]:sub1_coords[1] + sub1_coords[4], sub1_coords[2]:sub1_coords[2] + sub1_coords[5]]

        image_weak = image_weak[sub2_coords[0]:sub2_coords[0] + sub2_coords[3], sub2_coords[1]:sub2_coords[1] + sub2_coords[4], sub2_coords[2]:sub2_coords[2] + sub2_coords[5]]
        image_strong = image_strong[sub2_coords[0]:sub2_coords[0] + sub2_coords[3], sub2_coords[1]:sub2_coords[1] + sub2_coords[4], sub2_coords[2]:sub2_coords[2] + sub2_coords[5]]
        label = label[sub2_coords[0]:sub2_coords[0] + sub2_coords[3], sub2_coords[1]:sub2_coords[1] + sub2_coords[4], sub2_coords[2]:sub2_coords[2] + sub2_coords[5]]

        
        return {'image_weak': image_weak, 'image_strong': image_strong, 'label': label, 'cons_image_weak': cons_image_weak, 'cons_image_strong': cons_image_strong, 'cons_label': cons_label, 
                 'sub1_coords' : sub1_coords, 'sub2_coords' : sub2_coords, 'overlap_coords_in_sub1' : overlap_coords_in_sub1, 'overlap_coords_in_sub2' : overlap_coords_in_sub2, 
                 'normal_range_x' : (overlap_coords_in_sub2[0], overlap_coords_in_sub2[0] + overlap_coords_in_sub2[3]), 
                 'cons_range_x' : (overlap_coords_in_sub1[0], overlap_coords_in_sub1[0] + overlap_coords_in_sub1[3]), 
                 'normal_range_y' : (overlap_coords_in_sub2[1], overlap_coords_in_sub2[1] + overlap_coords_in_sub2[4]), 
                 'cons_range_y' : (overlap_coords_in_sub1[1], overlap_coords_in_sub1[1] + overlap_coords_in_sub1[4]), 
                 'normal_range_z' : (overlap_coords_in_sub2[2], overlap_coords_in_sub2[2] + overlap_coords_in_sub2[5]), 
                 'cons_range_z' : (overlap_coords_in_sub1[2], overlap_coords_in_sub1[2] + overlap_coords_in_sub1[5]), 
                 }

class RandomCrop_PCT(object):
    """
    Crop randomly the data in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_weak, image_strong, label = sample['image_weak'], sample['image_strong'], sample['label']
        # pad the sample if necessary

        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = 0
            image_weak = np.pad(image_weak, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image_strong = np.pad(image_strong, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image_weak.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        cons_start_x = np.random.randint(0, w1) if w1 != 0 else w1
        cons_start_y = np.random.randint(0, h1) if h1 != 0 else h1
        cons_start_z = np.random.randint(0, d1) if d1 != 0 else d1

        # no-overlap issues
        cons_start_x = w1 - int(96/2) if w1 - cons_start_x > 96 else cons_start_x
        cons_start_y = h1 - int(96/2) if h1 - cons_start_y > 96 else cons_start_y
        cons_start_z = d1 - int(96/2) if d1 - cons_start_z > 96 else cons_start_z

        cons_weak_image = image_weak[cons_start_x:cons_start_x + self.output_size[0],
                     cons_start_y:cons_start_y + self.output_size[1],
                     cons_start_z:cons_start_z + self.output_size[2]]
        cons_strong_image = image_strong[cons_start_x:cons_start_x + self.output_size[0],
                     cons_start_y:cons_start_y + self.output_size[1],
                     cons_start_z:cons_start_z + self.output_size[2]]

        cons_label = label[cons_start_x:cons_start_x + self.output_size[0],
                     cons_start_y:cons_start_y + self.output_size[1],
                     cons_start_z:cons_start_z + self.output_size[2]]

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_weak = image_weak[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_strong = image_strong[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        # print(d1, self.output_size[2], cons_start_z)
        assert cons_weak_image.shape == image_weak.shape, print(cons_weak_image.shape, image_weak.shape)
        assert cons_label.shape == label.shape, print(cons_label.shape, label.shape)

        a = image_weak[0 if cons_start_x < w1 else cons_start_x-w1:96-(w1-cons_start_x) if cons_start_x < w1 else 96,
            0 if cons_start_y < h1 else cons_start_y-h1:96-(h1-cons_start_y) if cons_start_y < h1 else 96,
            0 if cons_start_z < d1 else cons_start_z-d1:96-(d1-cons_start_z) if cons_start_z < d1 else 96]

        b = cons_weak_image[0 if cons_start_x > w1 else w1-cons_start_x:96-(cons_start_x-w1) if cons_start_x > w1 else 96,
            0 if cons_start_y > h1 else h1-cons_start_y:96-(cons_start_y-h1) if cons_start_y > h1 else 96,
            0 if cons_start_z > d1 else d1-cons_start_z:96-(cons_start_z-d1) if cons_start_z > d1 else 96]

        assert np.all(np.equal(a, b)), "?"
        return {'image_weak': image_weak, 'image_strong': image_strong, 'label': label, 'cons_weak_image': cons_weak_image, 'cons_strong_image': cons_strong_image, 'cons_label': cons_label,
                'normal_range_x': [0 if cons_start_x < w1 else cons_start_x-w1, 96-(w1-cons_start_x) if cons_start_x < w1 else 96],
                'cons_range_x': [0 if cons_start_x > w1 else w1-cons_start_x, 96-(cons_start_x-w1) if cons_start_x > w1 else 96],
                'normal_range_y': [0 if cons_start_y < h1 else cons_start_y-h1, 96-(h1-cons_start_y) if cons_start_y < h1 else 96],
                'cons_range_y': [0 if cons_start_y > h1 else h1-cons_start_y, 96-(cons_start_y-h1) if cons_start_y > h1 else 96,],
                'normal_range_z': [0 if cons_start_z < d1 else cons_start_z-d1, 96-(d1-cons_start_z) if cons_start_z < d1 else 96],
                'cons_range_z': [0 if cons_start_z > d1 else d1-cons_start_z, 96-(cons_start_z-d1) if cons_start_z > d1 else 96]}

class ToTensor_PCT(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image_weak = sample['image_weak']
        image_weak = image_weak.reshape(1, image_weak.shape[0], image_weak.shape[1], image_weak.shape[2]).astype(np.float32)
        image_strong = sample['image_strong']
        image_strong = image_strong.reshape(1, image_strong.shape[0], image_strong.shape[1], image_strong.shape[2]).astype(np.float32)

        cons_image_weak = sample['cons_weak_image']
        cons_image_weak = cons_image_weak.reshape(1, cons_image_weak.shape[0], cons_image_weak.shape[1], cons_image_weak.shape[2]).astype(np.float32)
        cons_image_strong = sample['cons_strong_image']
        cons_image_strong = cons_image_strong.reshape(1, cons_image_strong.shape[0], cons_image_strong.shape[1], cons_image_strong.shape[2]).astype(np.float32)
        
        a = {'image_weak': torch.from_numpy(image_weak), 'image_strong': torch.from_numpy(image_strong), 'label': torch.from_numpy(sample['label']).long(), 
                'x_range': sample['normal_range_x'], 'y_range': sample['normal_range_y'], 'z_range': sample['normal_range_z'],}
        b = {'image_weak': torch.from_numpy(cons_image_weak), 'image_strong': torch.from_numpy(cons_image_strong), 'label': torch.from_numpy(sample['cons_label']).long(), 
                'x_range': sample['cons_range_x'], 'y_range': sample['cons_range_y'], 'z_range': sample['cons_range_z'], } 
        return a, b

        # image = sample['data']
        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        # cons_image = sample['cons_image']
        # cons_image = cons_image.reshape(1, cons_image.shape[0], cons_image.shape[1], cons_image.shape[2]).astype(np.float32)
        # return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
        #         'x_range': sample['normal_range_x'], 'y_range': sample['normal_range_y'],
        #         'z_range': sample['normal_range_z']}, \
        #        {'data': torch.from_numpy(cons_image), 'label': torch.from_numpy(sample['cons_label']).long(),
        #         'x_range': sample['cons_range_x'], 'y_range': sample['cons_range_y'], 'z_range': sample['cons_range_z']}

class RandomRotFlip_consistency(object):
    """
    Randomly rotate and flip the image and label in a sample.
    Args:
        None
    """

    def __call__(self, image):
        k = random.randint(0, 3)  # Randomly choose rotation count
        dims_ = random.choice([(2, 3), (2, 4), (3, 4)])
        image = torch.rot90(image, k, dims=dims_)  # Rotate image
        axis = random.randint(2, 4)  # Randomly choose flip axis
        # print(dims_, axis)
        image = torch.flip(image, dims=(axis,))  # Flip image
        return image.clone(), k, dims_, axis

# 注意，这里是一个二维操作，应该要转化为三维的
class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2], image.shape[3], image.shape[4]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}

class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class ToTensor_o(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image_weak = sample['image_weak']
        image_weak = image_weak.reshape(1, image_weak.shape[0], image_weak.shape[1], image_weak.shape[2]).astype(np.float32)
        image_strong = sample['image_strong']
        image_strong = image_strong.reshape(1, image_strong.shape[0], image_strong.shape[1], image_strong.shape[2]).astype(np.float32)

        cons_image_weak = sample['cons_image_weak']
        cons_image_weak = cons_image_weak.reshape(1, cons_image_weak.shape[0], cons_image_weak.shape[1], cons_image_weak.shape[2]).astype(np.float32)
        cons_image_strong = sample['cons_image_strong']
        cons_image_strong = cons_image_strong.reshape(1, cons_image_strong.shape[0], cons_image_strong.shape[1], cons_image_strong.shape[2]).astype(np.float32)

        a = {'image_weak': torch.from_numpy(image_weak), 'image_strong': torch.from_numpy(image_strong), 'label': torch.from_numpy(sample['label'].astype(np.int32)).long(), 
                'x_range': sample['normal_range_x'], 'y_range': sample['normal_range_y'], 'z_range': sample['normal_range_z'], 
                'sub2_coords' : sample['sub2_coords'], 'overlap_coords_in_sub2' : sample['overlap_coords_in_sub2']}
        b = {'image_weak': torch.from_numpy(cons_image_weak), 'image_strong': torch.from_numpy(cons_image_strong), 'label': torch.from_numpy(sample['cons_label'].astype(np.int32)).long(), 
                'x_range': sample['cons_range_x'], 'y_range': sample['cons_range_y'], 'z_range': sample['cons_range_z'], 
                'sub1_coords' : sample['sub1_coords'], 'overlap_coords_in_sub1' : sample['overlap_coords_in_sub1']} 
        return a, b
    
class Normalise(object):
    def __call__(self, sample):
        image = sample['image']
        return{'image': (image - image.min()) / (image.max()-image.min()), 'label': sample['label']}

class RandomBrightness(object):
    def __init__(self, region):
        self.region = region

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        scale = random.uniform(self.region[0], self.region[1])
        image = image * scale
        image = np.clip(image, a_min=0., a_max=1.)
        sample = {'image': image, 'label': label}
        return sample
    
class RandomGaussianNoise(object):
    def __init__(self, sigma=[0.1, 1.0], apply_prob=0.6):
        self.s_min, self.s_max = min(sigma), max(sigma)
        self.prob = apply_prob

    def __call__(self, image):
        # image = sample['image']
        # print(image.shape)
        self.sigma = np.random.uniform(self.s_min, self.s_max)

        if np.random.uniform() < self.prob:
            noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)  # 注意这儿真有问题，指不定真要放到dataset里面去
            image = image + noise
        # return {'image': image, 'label': sample['label']}
        return image
    
MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


def preserve_shape(func):
    """
    Preserve shape of the image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function

@clipped
def _brightness_contrast_adjust_non_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = img.dtype
    img = img.astype("float32")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


@preserve_shape
def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = np.dtype("uint8")

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    lut = np.arange(0, max_value + 1).astype("float32")

    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += beta * np.mean(img)

    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img

def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):
    if img.dtype == np.uint8:
        return _brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)

    return _brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)
    
class RandomBrightnessContrast(object):
    def __init__(self, 
                 brightness_limit=0.5,
                 contrast_limit=0.5,
                 prob=0.8):
        assert 0<=brightness_limit<=1
        assert 0<=contrast_limit<=1
        assert 0<=prob<=1
        
        self.contrast_limit = contrast_limit
        self.brightness_limit = brightness_limit
        
        self.alpha = 1.0
        self.beta = 0.0
        self.prob = prob
    
    def _random_update(self):
        self.alpha = 1.0 + np.random.uniform(-1.0 * self.contrast_limit, self.contrast_limit),
        self.beta = 0.0 + np.random.uniform(-1.0 * self.brightness_limit, self.brightness_limit)
        

    def __call__(self, image):
        # image = sample['image']
        image = image.astype(np.float32)
        self._random_update()
        if np.random.uniform() < self.prob:
            img_min, img_max = image.min(), image.max()
            image_norm = (image - img_min) / (img_max - img_min)
            image_norm = brightness_contrast_adjust(image_norm, alpha=self.alpha, beta=self.beta)
            image = image_norm * (img_max - img_min) + img_min

        # return {'image': image, 'label': sample['label']}
        return image