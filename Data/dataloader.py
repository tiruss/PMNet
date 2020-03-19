from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch
import numpy as np


class Resize(object):
    def __init__(self, size, train):
        self.size = size
        self.train = train

    def __call__(self, sample):
        if self.train:
            img, mask, contour = sample["image"], sample["mask"], sample['contour']
            img, mask, contour = img.resize((self.size, self.size), resample=Image.BILINEAR),\
                                 mask.resize((self.size, self.size), resample=Image.BILINEAR),\
                                 contour.resize((self.size, self.size), resample=Image.BILINEAR)

            sample = {"image":img, 'mask':mask, 'contour':contour}

            return sample
        else:
            img, mask = sample["image"], sample["mask"]

            img, mask = img.resize((self.size, self.size), resample=Image.BILINEAR),\
                        mask.resize((self.size, self.size), resample=Image.BILINEAR)

            sample = {'image':img, 'mask':mask}

            return sample

class Down(object):
    def __init__(self, size, train):
        self.size = size
        self.train = train

    def __call__(self, sample):
        if self.train:
            img, mask, contour = sample['image'], sample['mask'], sample['contour']
            mask = mask.resize((self.size, self.size))
            contour = contour.resize((self.size, self.size))

            return {'image':img, 'mask':mask, 'contour':contour}
        else:
            img, mask = sample['image'], sample['mask']
            mask = mask.resize((self.size, self.size))

            return {'image':img, 'mask':mask}

class RandomCrop(object):
    def __init__(self, size, train):
        self.train = train
        self.size = size

    def __call__(self, sample):
        global img, mask, contour

        if self.train:
            img, mask, contour = sample["image"], sample["mask"], sample['contour']
            img, mask, contour = img.resize((256, 256), resample=Image.BILINEAR),\
                                 mask.resize((256, 256), resample=Image.BILINEAR),\
                                 contour.resize((256, 256), resample=Image.BILINEAR)

            h, w = img.size
            new_h, new_w = self.size, self.size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            img = img.crop((left, top, left + new_w, top + new_h))
            mask = mask.crop((left, top, left + new_w, top + new_h))
            contour = contour.crop((left, top, left + new_w, top + new_h))

            return {'image': img, 'mask': mask, 'contour': contour}
        else:
            img, mask  = sample["image"], sample["mask"]
            img, mask = img.resize((256, 256), resample=Image.BILINEAR),\
                        mask.resize((256, 256), resample=Image.BILINEAR),\

            h, w = img.size
            new_h, new_w = self.size, self.size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            img = img.crop((left, top, left+new_w, top+new_h))
            mask = mask.crop((left, top, left+new_w, top+new_h))

            return {'image':img, 'mask':mask}


class RandomFlip(object):
    def __init__(self, prob, train):
        self.train = train
        self.prob = prob
        self.flip = transforms.RandomHorizontalFlip(1.)

    def __call__(self, sample):
        if self.train:
            if np.random.random_sample() < self.prob:
                img, mask, contour = sample['image'], sample['mask'], sample['contour']
                img, mask, contour = self.flip(img), self.flip(mask), self.flip(contour)

                return {'image':img, 'mask':mask, 'contour':contour}
            else:
                return sample
        else:
            if np.random.random_sample() < self.prob:
                img, mask = sample['image'], sample['mask']
                img, mask = self.flip(img), self.flip(mask)

                return {'image': img, 'mask': mask}
            else:
                return sample

class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        if self.train:
            img, mask, contour = sample['image'], sample['mask'], sample['contour']
            img, mask, contour = self.tensor(img), self.tensor(mask), self.tensor(contour)

            return {'image':img, 'mask':mask, 'contour':contour}
        else:
            img, mask = sample['image'], sample['mask']
            img, mask = self.tensor(img), self.tensor(mask)

            return {'image': img, 'mask': mask}

class custom_dataloader(Dataset):
    def __init__(self, img_dir, mask_dir, contour_dir=None, train=False, down_scale=4):
        self.train = train
        self.down_scale = down_scale
        self.img_list = sorted(glob(img_dir + "*"))
        self.mask_list = sorted(glob(mask_dir + "*.png"))
        if self.train:
            self.contour_list = sorted(glob(contour_dir + "*"))

        if self.train:
            self.img_transform = transforms.Compose([
                Resize(256, train=self.train),
                RandomFlip(0.5, train=self.train),
                RandomCrop(224, train=self.train),
                Down(int(224 / pow(2, self.down_scale)), train=self.train),
                ToTensor(train=self.train),
            ])
        else:
            self.img_transform = transforms.Compose([
                Resize(224, train=self.train),
                ToTensor(train=self.train)
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        mask_name = self.mask_list[item]

        img = Image.open(img_name)
        mask = Image.open(mask_name)

        w, h = img.size

        name = self.img_list[item].split("\\")[-1]

        if self.train:
            img = img.convert('RGB')
            mask = mask.convert('L')

            contour_name = self.contour_list[item]
            contour = Image.open(contour_name)
            contour = contour.convert('L')
            sample = {'image': img, 'mask': mask, 'contour': contour}

            sample = self.img_transform(sample)

            return sample, name, (w, h)
        else:
            img = img.convert('RGB')
            mask = mask.convert('L')

            orig_mask = mask.copy()

            sample = {'image': img, 'mask': mask}

            sample = self.img_transform(sample)

            return sample, name, (w, h)